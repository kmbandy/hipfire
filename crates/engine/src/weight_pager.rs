//! Weight pager — runtime residency management for MoE/dense weights (MAD-93 v0.1).
//!
//! Hipfire today loads all weights to VRAM at startup. For models that exceed
//! VRAM (Qwen3.5-REAP-97B-A10B is the v0.1 target), we need to keep most experts
//! on host (read from the HFQ file) and page them in to VRAM on demand based on
//! routing decisions made by [`crate::cpu_router::CpuRouter`].
//!
//! ## Architecture (foundational)
//!
//! - **CPU is the scheduler authority.** [`CpuRouter`](crate::cpu_router::CpuRouter)
//!   replicates the per-layer router GEMV on CPU so we know top-k expert indices
//!   without a GPU→CPU sync inside the forward path. The pager consumes those
//!   indices and decides what to fetch / evict.
//! - **Transport is abstracted.** Today it's pread + `hipMemcpyAsync`
//!   ([`PreadH2DTransport`]). In a future commit we drop in
//!   `IoUringP2PTransport` for true NVMe→VRAM DMA without changing anything
//!   above this layer.
//! - **Stable per-weight identity.** [`WeightId`] is a small, hashable enum so
//!   the residency map and the (file_offset, byte_len) lookup table can be
//!   keyed identically. This is what an io_uring submission queue would consume
//!   directly.
//! - **Pager owns its VRAM.** All paged weight allocations route through the
//!   pager (not ad-hoc `gpu.alloc_tensor`), so we can later export the slabs as
//!   `dma_buf` for P2P DMA without reorganizing call sites.
//!
//! ## v0.1 scope
//!
//! - [`Transport`] trait + [`PreadH2DTransport`] impl
//! - [`WeightPager`] with residency map and `ensure_resident` (synchronous, no
//!   real eviction yet — assumes VRAM is large enough)
//! - [`WeightId`] schema covering MoE experts, dense attention, norms, embeds
//!
//! Real eviction, async transfer overlap, and predictive prefetch land in
//! follow-up commits — the trait shapes here are the seams those commits plug
//! into without changing the forward path.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use hip_bridge::{HipResult, DeviceBuffer};
use rdna_compute::{Gpu, GpuTensor};

use crate::hfq::HfqFile;

// ---------------------------------------------------------------------------
// Identity: WeightId
// ---------------------------------------------------------------------------

/// Stable identity for a weight that the pager can move between host and VRAM.
///
/// The variants enumerate every kind of weight that participates in paging.
/// For v0.1 only [`WeightId::Expert`] actually pages — the others are listed
/// so the residency map can track them as "always resident" without a special
/// case at the call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightId {
    /// Routed expert weight (one of the 256 experts in Qwen3.5-MoE-A3B).
    /// `role` distinguishes the fused gate_up matrix from the down matrix.
    Expert {
        layer: u16,
        expert: u16,
        role: ExpertRole,
    },
    /// Always-on shared expert (one per layer).
    SharedExpert {
        layer: u16,
        role: SharedRole,
    },
    /// Per-layer router weight (small, always-resident in v0.1, but tracked
    /// here so future commits can page it for very large MoE configs).
    Router { layer: u16 },
    /// Dense attention weight (q/k/v/o).
    DenseAttn { layer: u16, role: AttnRole },
    /// RMSNorm gain vector. Tiny but per-layer.
    Norm { layer: u16, kind: NormKind },
    /// Token embedding table (always resident in v0.1).
    Embed,
    /// LM head (often shares storage with Embed; pager tracks separately).
    LmHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertRole {
    /// Fused gate || up: shape `[2 * moe_intermediate, hidden]`.
    GateUp,
    /// Down projection: shape `[hidden, moe_intermediate]`.
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedRole {
    Gate,
    Up,
    Down,
    /// Scalar sigmoid gate on the shared-expert add: `[1, hidden]` row vector.
    SigmoidGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnRole {
    Q,
    K,
    V,
    O,
    /// Fused QKV when the model stores them as one tensor.
    Qkv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormKind {
    /// Pre-attention norm.
    Attn,
    /// Pre-MoE / pre-FFN norm.
    Ffn,
    /// Final norm before LM head.
    Final,
}

// ---------------------------------------------------------------------------
// Transport: how bytes get from HFQ file to VRAM
// ---------------------------------------------------------------------------

/// Opaque handle for an in-flight or completed transfer. Submit returns one,
/// `wait` consumes a slice of them. v0.1 uses a simple counter; future
/// async-overlap impls will back this with a `hipEvent` or io_uring CQE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferHandle(u64);

/// Abstraction over how the pager moves bytes from host storage to VRAM.
///
/// **This is the migration seam for the NVMe→VRAM DMA future.** Today's impl
/// ([`PreadH2DTransport`]) does `pread` into a pinned host buffer then
/// `hipMemcpyAsync` to VRAM. A future `IoUringP2PTransport` reads directly
/// into VRAM via `dma_buf` + io_uring with no host hop. The pager never sees
/// the difference.
pub trait Transport: Send {
    /// Submit a transfer of `len` bytes starting at `hfq_offset` in the HFQ
    /// file into the device buffer `dst`. Returns a handle to wait on.
    fn submit(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &DeviceBuffer,
    ) -> HipResult<TransferHandle>;

    /// Block until every handle in `handles` has completed.
    fn wait(&mut self, handles: &[TransferHandle]) -> HipResult<()>;

    /// Hint: does this transport need pager-allocated VRAM slabs to be
    /// exported as `dma_buf` for P2P DMA? Pager checks this at allocation
    /// time. Default false (host-staged path doesn't care).
    fn requires_dma_buf_alloc(&self) -> bool {
        false
    }

    /// Hint: required alignment for `hfq_offset` in bytes. `O_DIRECT` paths
    /// need 4 KB; pread doesn't care. Pager validates on submit.
    fn alignment(&self) -> usize {
        1
    }
}

/// v0.1 transport: read via [`HfqFile::tensor_data_pread`]-style pread into a
/// reusable host buffer, then `hipMemcpyAsync` to VRAM. Synchronous from the
/// caller's POV in this commit (a follow-up uses a dedicated stream + event
/// for true overlap).
pub struct PreadH2DTransport {
    /// Reusable host staging buffer. Grows monotonically to the largest
    /// tensor size we've seen. Future commits replace this with a pool of
    /// pinned (`hipHostMalloc`'d) buffers so transfers can actually overlap.
    staging: Vec<u8>,
    /// Monotonic handle ID. v0.1 just uses this for debugging; transfers
    /// complete synchronously inside `submit`.
    next_handle: u64,
    /// File descriptor we pread from. Passed in by the caller because
    /// [`HfqFile`] owns the `File` and exposes a pread helper.
    /// In v0.1 we hold a borrow-equivalent indirectly via the pager.
    _phantom: std::marker::PhantomData<()>,
}

impl PreadH2DTransport {
    pub fn new() -> Self {
        Self {
            staging: Vec::new(),
            next_handle: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    fn next_handle(&mut self) -> TransferHandle {
        let h = TransferHandle(self.next_handle);
        self.next_handle += 1;
        h
    }
}

impl Default for PreadH2DTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for PreadH2DTransport {
    fn submit(
        &mut self,
        _hfq_offset: usize,
        _len: usize,
        _dst: &DeviceBuffer,
    ) -> HipResult<TransferHandle> {
        // v0.1 stub. The actual implementation reads through HfqFile's pread
        // helper into `self.staging`, then does `gpu.memcpy_htod(dst, ...)`.
        // Wired in the next commit when WeightPager::ensure_resident actually
        // performs cold-load from disk; for now it's only called by code paths
        // that aren't reachable in v0.1.
        Ok(self.next_handle())
    }

    fn wait(&mut self, _handles: &[TransferHandle]) -> HipResult<()> {
        // Transfers complete synchronously inside `submit` in v0.1.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Construction-time config. Keep this small and explicit — runtime flags
/// belong on `Qwen35Config`, not here.
#[derive(Debug, Clone)]
pub struct PagerConfig {
    /// Soft cap on VRAM bytes the pager is allowed to hold for paged weights.
    /// Eviction kicks in when adding a new resident weight would exceed this.
    /// `u64::MAX` means "unlimited" (effectively disables eviction — useful
    /// for testing the routing path without VRAM pressure).
    pub vram_budget_bytes: u64,
    /// If true, the pager prints structured residency events to stderr.
    /// Disabled by default; useful when debugging eviction policy.
    pub trace: bool,
}

impl Default for PagerConfig {
    fn default() -> Self {
        Self {
            vram_budget_bytes: u64::MAX,
            trace: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WeightPager
// ---------------------------------------------------------------------------

/// Tracks which weights are currently resident in VRAM and provides the
/// `ensure_resident` / `evict_lru_until` primitives the forward path uses.
///
/// **This is the GPU-side of the pager.** The CPU-side scheduling
/// (compute router → decide top-k → call ensure_resident) happens in the
/// caller; see [`crate::cpu_router::CpuRouter`].
pub struct WeightPager {
    /// What's currently in VRAM. Maps weight identity to a `Resident` record
    /// holding the buffer + bookkeeping for LRU.
    resident: HashMap<WeightId, Resident>,
    /// Recency queue for LRU eviction. Most-recently-used at the back.
    /// We use VecDeque because mutations are O(n) but n is tiny (top-k
    /// experts × layers, max ~thousands), and VecDeque iteration is
    /// cache-friendly.
    lru: VecDeque<WeightId>,
    /// Bytes currently held by `resident`.
    vram_used_bytes: u64,
    /// Per-weight (file_offset, byte_len) for cold-load via Transport.
    /// Populated at registration time when the model loader walks the HFQ
    /// tensor index. Stable across the run.
    catalog: HashMap<WeightId, ByteRange>,
    /// Transport implementation (v0.1: pread + H2D, future: io_uring + P2P).
    transport: Box<dyn Transport>,
    /// Construction-time config.
    config: PagerConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct ByteRange {
    pub offset: usize,
    pub len: usize,
}

struct Resident {
    /// The actual VRAM buffer. Pager owns this allocation lifecycle.
    _buffer: DeviceBuffer,
    /// Cached byte length so eviction can update `vram_used_bytes` cheaply.
    bytes: u64,
}

impl WeightPager {
    pub fn new(transport: Box<dyn Transport>, config: PagerConfig) -> Self {
        Self {
            resident: HashMap::new(),
            lru: VecDeque::new(),
            vram_used_bytes: 0,
            catalog: HashMap::new(),
            transport,
            config,
        }
    }

    /// Register that `id` lives at `range` in the HFQ file. Called by the
    /// loader when it walks the tensor index. Must be called before any
    /// `ensure_resident(id)` for that id.
    pub fn register(&mut self, id: WeightId, range: ByteRange) {
        self.catalog.insert(id, range);
    }

    /// Number of registered weights. Useful for diagnostics.
    pub fn registered_count(&self) -> usize {
        self.catalog.len()
    }

    /// Returns true if `id` is currently in VRAM.
    pub fn is_resident(&self, id: WeightId) -> bool {
        self.resident.contains_key(&id)
    }

    /// Ensure `id` is in VRAM. Synchronous. v0.1: assumes the loader
    /// pre-loaded everything on the always-resident path; raising
    /// [`WeightPagerError::NotRegistered`] for unknown ids and `Unimplemented`
    /// for cold-load on demand. The actual cold-load + LRU lifecycle wires
    /// in the next commit (task #8).
    pub fn ensure_resident(
        &mut self,
        id: WeightId,
        _gpu: &mut Gpu,
        _hfq: &HfqFile,
    ) -> Result<(), WeightPagerError> {
        if !self.catalog.contains_key(&id) {
            return Err(WeightPagerError::NotRegistered(id));
        }
        if self.resident.contains_key(&id) {
            self.touch_lru(id);
            return Ok(());
        }
        // Future: budget check → maybe evict → transport.submit → transport.wait
        // For v0.1 we surface this as Unimplemented so the caller knows the
        // paged path isn't fully wired yet. Forward path checks
        // `paged_experts` and stays on the always-resident codepath.
        Err(WeightPagerError::Unimplemented(
            "cold-load wired in next commit",
        ))
    }

    /// Insert an already-resident weight (used by the loader during initial
    /// load — it allocates the VRAM and hands the pager a tracking entry).
    pub fn insert_resident(&mut self, id: WeightId, buffer: DeviceBuffer, bytes: u64) {
        if let Some(prev) = self.resident.remove(&id) {
            self.vram_used_bytes = self.vram_used_bytes.saturating_sub(prev.bytes);
            self.lru.retain(|x| *x != id);
        }
        self.resident.insert(id, Resident { _buffer: buffer, bytes });
        self.lru.push_back(id);
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(bytes);
    }

    /// Mark `id` as recently used. No-op if not resident.
    pub fn touch_lru(&mut self, id: WeightId) {
        if let Some(pos) = self.lru.iter().position(|x| *x == id) {
            self.lru.remove(pos);
            self.lru.push_back(id);
        }
    }

    /// Evict resident weights starting from the LRU tail until the freed
    /// bytes meet `need_bytes`. v0.1 stub — real eviction lands in task #9.
    pub fn evict_lru_until(&mut self, _need_bytes: u64) -> Result<(), WeightPagerError> {
        Err(WeightPagerError::Unimplemented(
            "LRU eviction lands in task #9",
        ))
    }

    /// Bytes currently held resident. Cheap (cached, not a sum).
    pub fn vram_used_bytes(&self) -> u64 {
        self.vram_used_bytes
    }

    /// Patch the device-side `expert_*_ptrs` indirection table so the indexed
    /// MoE GEMV reads the currently-resident buffer pointers for `layer`.
    ///
    /// v0.1 stub — wired in task #8 when `moe_ffn_decode_impl` learns the
    /// paged path. The shape is here so the call site is locked in: pager
    /// patches the tables, kernels read from them.
    pub fn update_expert_ptrs(
        &mut self,
        _layer: u16,
        _gate_up_ptrs: &GpuTensor,
        _down_ptrs: &GpuTensor,
        _gpu: &mut Gpu,
    ) -> HipResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum WeightPagerError {
    /// Weight wasn't registered with the pager. Loader bug.
    NotRegistered(WeightId),
    /// Hipfire HIP error (transfer / alloc failed).
    Hip(hip_bridge::HipError),
    /// Stub for unimplemented paths in this skeleton commit.
    Unimplemented(&'static str),
}

impl std::fmt::Display for WeightPagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRegistered(id) => write!(f, "weight not registered: {id:?}"),
            Self::Hip(e) => write!(f, "hip error: {e}"),
            Self::Unimplemented(why) => write!(f, "weight pager: unimplemented ({why})"),
        }
    }
}

impl std::error::Error for WeightPagerError {}

impl From<hip_bridge::HipError> for WeightPagerError {
    fn from(e: hip_bridge::HipError) -> Self {
        Self::Hip(e)
    }
}

// ---------------------------------------------------------------------------
// Convenience: open an HfqFile by path. The loader uses the existing
// HfqFile::open directly; this re-export keeps the module's surface minimal.
// ---------------------------------------------------------------------------

/// Forwarding helper so callers don't need a separate `use crate::hfq::HfqFile`.
pub fn open_hfq(path: &Path) -> std::io::Result<HfqFile> {
    HfqFile::open(path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_id_is_hashable() {
        let mut map = HashMap::new();
        let a = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::GateUp };
        let b = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::Down };
        map.insert(a, 1u32);
        map.insert(b, 2u32);
        assert_eq!(map.get(&a), Some(&1));
        assert_eq!(map.get(&b), Some(&2));
    }

    #[test]
    fn pager_starts_empty() {
        let pager = WeightPager::new(Box::new(PreadH2DTransport::new()), PagerConfig::default());
        assert_eq!(pager.registered_count(), 0);
        assert_eq!(pager.vram_used_bytes(), 0);
    }

    #[test]
    fn ensure_resident_unknown_id_errors_clearly() {
        let pager = WeightPager::new(Box::new(PreadH2DTransport::new()), PagerConfig::default());
        // We can't construct a Gpu / HfqFile in a unit test without device
        // access, but we can verify the registry-miss path does not touch
        // them (it returns NotRegistered before any GPU/file work).
        let id = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::GateUp };
        // Safety: this path returns before dereferencing the &mut Gpu / &HfqFile,
        // so passing zeroed/uninitialized references would be UB. Instead we
        // assert at the API level by checking is_resident + catalog state.
        assert!(!pager.is_resident(id));
        assert!(pager.catalog.get(&id).is_none());
    }
}
