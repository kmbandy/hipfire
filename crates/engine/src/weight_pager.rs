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
use std::fs::File;
use std::path::Path;

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

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
    /// Dense FFN weight (gate / up / down). The big-three for paged dense
    /// inference — Qwen3.6-27B's per-layer FFN is ~140 MB at MQ4 (across
    /// all three roles), 64 layers ≈ 9 GB. Paging unlocks 27B Q4 on 16 GB
    /// GPUs (MAD-94 v0.2).
    DenseFfn { layer: u16, role: FfnRole },
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

/// Dense FFN weight role. The standard transformer FFN is
/// `down(silu(gate) * up)` over `[hidden]→[intermediate]→[hidden]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FfnRole {
    /// Gate projection: `[intermediate, hidden]`.
    Gate,
    /// Up projection: `[intermediate, hidden]`.
    Up,
    /// Down projection: `[hidden, intermediate]`.
    Down,
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

/// Index into `WeightPager::transfer_events`. Returned by
/// `fill_into_async`; consumed by `wait_transfer`. v0.3-β scope —
/// the slot is single-shot, calling wait twice on the same idx is a
/// usage bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferEventIdx(usize);

/// Abstraction over how the pager moves bytes from host storage to VRAM.
///
/// **This is the migration seam for the NVMe→VRAM DMA future.** Today's impl
/// ([`PreadH2DTransport`]) does `pread` into a host staging buffer then
/// `hipMemcpyAsync` to VRAM. A future `IoUringP2PTransport` reads directly
/// into VRAM via `dma_buf` + io_uring with no host hop. The pager never sees
/// the difference.
///
/// `fetch` is responsible for both the allocation (so io_uring can use a
/// `dma_buf`-exportable slab when needed) and the transfer. `wait` exists for
/// future async overlap; today's pread path completes synchronously inside
/// `fetch` and `wait` is a no-op.
pub trait Transport: Send {
    /// Allocate a `GpuTensor` (with `DType::Raw`, shape `[len]`) and populate
    /// it with `len` bytes from `hfq_offset` in the HFQ file. Returns the
    /// fresh tensor and a handle that can be waited on.
    ///
    /// In v0.1 the transfer is synchronous (handle is informational); in
    /// follow-ups the transport may submit async and have callers `wait`.
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)>;

    /// Copy `len` bytes from `hfq_offset` into the **existing** device
    /// buffer `dst`. Used by the scratch-buffer paging pattern (dense
    /// paging) where one fixed-size buffer is reused across layers
    /// rather than allocating a fresh one per page-in. Caller guarantees
    /// `dst.buf.size >= len`.
    fn fill_into(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &GpuTensor,
        gpu: &mut Gpu,
    ) -> HipResult<()>;

    /// Read `len` bytes from `hfq_offset` directly into the caller's
    /// host slice (typically pinned via `hipHostMalloc`). No GPU work —
    /// the WeightPager v0.3 host tier uses this to populate pinned-RAM
    /// shadows of paged weights without allocating a device buffer.
    /// Caller guarantees `dst.len() == len`.
    fn read_to_host(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &mut [u8],
    ) -> HipResult<()>;

    /// Block until every handle in `handles` has completed. v0.1 no-op
    /// because `fetch` is synchronous; defined for forward compatibility.
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

/// v0.1 transport: pread the requested byte range from the HFQ file into a
/// reusable host buffer, then upload to VRAM via [`Gpu::upload_raw`]
/// (which internally does `hipMalloc` + `hipMemcpy(H2D)`).
///
/// Synchronous in this commit. A follow-up commit replaces the staging with a
/// pool of pinned (`hipHostMalloc`'d) buffers and uses `hipMemcpyAsync` on a
/// dedicated stream so the next-layer prefetch can overlap with current-layer
/// compute.
pub struct PreadH2DTransport {
    /// Owned file handle for the HFQ file. We open our own (rather than
    /// borrowing `HfqFile`'s) so a future `IoUringP2PTransport` can register
    /// its fd with io_uring + `dma_buf` independently. Path is held alongside
    /// for diagnostics.
    file: File,
    path: std::path::PathBuf,
    /// Reusable host staging buffer. Grows monotonically to the largest
    /// tensor size we've seen.
    staging: Vec<u8>,
    /// Monotonic handle ID. v0.1 fetches complete synchronously, so this is
    /// purely informational; future async impls will key real completion
    /// state on this id.
    next_handle: u64,
}

impl PreadH2DTransport {
    /// Open the HFQ file at `path` for paged reads.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // Hint sequential-ish access for the page-cache layer. Tensors don't
        // overlap so reads are effectively sequential within a tensor and
        // random across tensors; the kernel's readahead handles the within
        // case correctly with this advice.
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
            }
        }
        Ok(Self {
            file,
            path: path.to_path_buf(),
            staging: Vec::new(),
            next_handle: 0,
        })
    }

    /// Path the transport was opened with. Useful for diagnostics + the
    /// future io_uring impl which needs to register the same path with
    /// io_uring SQE buffers.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn next_handle(&mut self) -> TransferHandle {
        let h = TransferHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    /// Read `len` bytes at `offset` into `self.staging[..len]`. Linux uses
    /// `pread` (positional read, no seek state); other platforms fall back
    /// to `seek + read_exact` (correct but loses thread-safety on the file
    /// — a non-issue today since the pager is single-threaded).
    fn pread_into_staging(&mut self, offset: usize, len: usize) -> std::io::Result<()> {
        if self.staging.len() < len {
            self.staging.resize(len, 0);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut self.staging[..len], offset as u64)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(offset as u64))?;
            self.file.read_exact(&mut self.staging[..len])?;
        }
        Ok(())
    }
}

impl Transport for PreadH2DTransport {
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)> {
        // 1. Host: pread the bytes into our staging buffer.
        self.pread_into_staging(hfq_offset, len)
            .map_err(|e| {
                hip_bridge::HipError::new(0, &format!(
                    "pread {} bytes at offset {}: {}",
                    len, hfq_offset, e
                ))
            })?;
        // 2. GPU: alloc + memcpy_htod via the existing rdna-compute helper.
        //    `dtype: Raw` because the pager doesn't care about element layout
        //    — that interpretation belongs to `WeightTensor` at the call site.
        let tensor = gpu.upload_raw(&self.staging[..len], &[len])?;
        Ok((tensor, self.next_handle()))
    }

    fn fill_into(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &GpuTensor,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        // pread → staging, then memcpy_htod into the caller's existing buffer.
        // Used for dense paging where we have one set of scratch buffers
        // (gate/up/down) reused across layers — alloc-once, fill-many.
        self.pread_into_staging(hfq_offset, len)
            .map_err(|e| {
                hip_bridge::HipError::new(0, &format!(
                    "pread {} bytes at offset {} (fill_into): {}",
                    len, hfq_offset, e
                ))
            })?;
        gpu.hip.memcpy_htod(&dst.buf, &self.staging[..len])
    }

    fn read_to_host(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &mut [u8],
    ) -> HipResult<()> {
        // pread directly into the caller's slice — no staging, no GPU. The
        // dst slice typically backs a `hipHostMalloc`-pinned `HostBuffer`,
        // so this single read populates the host tier and primes future
        // promotions to fast PCIe copies.
        debug_assert_eq!(dst.len(), len);
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(dst, hfq_offset as u64)
                .map_err(|e| hip_bridge::HipError::new(0, &format!(
                    "pread {} bytes at offset {} (read_to_host): {}",
                    len, hfq_offset, e
                )))
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(hfq_offset as u64))
                .and_then(|_| self.file.read_exact(dst))
                .map_err(|e| hip_bridge::HipError::new(0, &format!(
                    "pread {} bytes at offset {} (read_to_host): {}",
                    len, hfq_offset, e
                )))
        }
    }

    fn wait(&mut self, _handles: &[TransferHandle]) -> HipResult<()> {
        // Transfers complete synchronously inside `fetch`/`fill_into` in v0.1.
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
    /// Soft cap on **pinned host RAM** bytes the pager may hold (v0.3 host
    /// tier). When > 0, weights cold-loaded from disk are also cached in
    /// pinned host memory; subsequent VRAM promotions copy host→device at
    /// PCIe bandwidth (~25 GB/s pinned vs ~5 GB/s pageable / NVMe). For
    /// the canonical "27B fits in RAM, partially in VRAM" use case, set
    /// this above the model's paged-weight footprint so the host tier
    /// never evicts and every page-in is a fast PCIe copy. `0` (default)
    /// disables the tier — behavior identical to v0.2 (every cold load
    /// re-reads from NVMe).
    pub host_budget_bytes: u64,
    /// If true, the pager prints structured residency events to stderr.
    /// Disabled by default; useful when debugging eviction policy.
    pub trace: bool,
}

impl Default for PagerConfig {
    fn default() -> Self {
        Self {
            vram_budget_bytes: u64::MAX,
            host_budget_bytes: 0,
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
    /// **Host (pinned-RAM) tier** (v0.3). Independent of `resident` — a
    /// weight may be only here, only in `resident`, in both (shadow), or
    /// neither (cold). When a vram-resident weight gets evicted, its host
    /// shadow stays so the next promotion is a cheap PCIe copy rather
    /// than another NVMe read. Empty when `config.host_budget_bytes == 0`.
    host_resident: HashMap<WeightId, HostResident>,
    /// One big shared arena for every host-resident weight. Allocated
    /// lazily on first prefetch — sized to `config.host_budget_bytes`.
    /// A contiguous bump arena is correct because v0.3-α only supports
    /// the "host budget ≥ total paged weights" regime (no eviction;
    /// warmup → permanent residency).
    ///
    /// **Unpinned** today (plain `Vec<u8>`) — pinned (`hipHostMalloc`)
    /// would unlock ~5× faster H2D copies (~25 GB/s vs ~5 GB/s on
    /// PCIe 4.0) but requires `RLIMIT_MEMLOCK` to be raised above the
    /// 8 MB Linux default, which most users haven't done. The host
    /// tier still wins because it skips the NVMe re-read on every
    /// page-in; pinned is the v0.3-β polish on top of that.
    host_arena: Option<Vec<u8>>,
    /// LRU queue for the host tier — kept for forward compatibility
    /// with v0.3-β eviction, but not consulted today.
    host_lru: VecDeque<WeightId>,
    /// Bytes currently held by `host_resident`. Doubles as the
    /// bump-arena cursor: next allocation lands at `host_arena[host_used_bytes..]`.
    host_used_bytes: u64,
    /// Dedicated GPU stream for async H2D transfers (v0.3-β). Distinct
    /// from the compute stream so transfer for layer N+1 can run
    /// concurrently with compute on layer N. Allocated lazily on first
    /// `fill_into_async` call so non-async callers don't pay for the
    /// stream creation.
    transfer_stream: Option<hip_bridge::Stream>,
    /// In-flight transfer events. `fill_into_async` records an event
    /// after each async memcpy and returns its index; `wait_transfer`
    /// syncs the event by index. Slots are reused — the LRU is just
    /// the index `Vec`, not a queue.
    transfer_events: Vec<hip_bridge::Event>,
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

/// Uniform-shape MoE expert metadata for v0.1.
///
/// Qwen3.5-MoE-A3B has 256 experts that all share the same gate_up and down
/// shape, so we store one set of dimensions per layer instead of per-expert.
/// When we add a heterogeneous-shape MoE arch (Mixtral derivatives, etc.),
/// this generalizes to `Vec<ExpertShape>` indexed by expert index.
#[derive(Debug, Clone, Copy)]
pub struct ExpertShape {
    /// Output rows of the fused gate_up matrix = `2 * moe_intermediate_size`.
    pub gate_up_m: usize,
    /// Input cols of gate_up = `hidden_size`.
    pub gate_up_k: usize,
    /// Output rows of down = `hidden_size`.
    pub down_m: usize,
    /// Input cols of down = `moe_intermediate_size`.
    pub down_k: usize,
    /// GPU dtype of the expert weights. Stashed here so the paged dispatch
    /// site can do dtype-based routing (MQ4 fast path vs mixed fallback)
    /// without reading from `ffn.experts[0]` — which is empty in paged mode.
    pub gpu_dtype: DType,
}

struct Resident {
    /// The actual VRAM tensor (pager owns its lifecycle). `dtype: Raw` —
    /// callers reinterpret the bytes via their own `WeightTensor` wrapper at
    /// access time. Storing as `GpuTensor` (rather than the lower-level
    /// `DeviceBuffer`) keeps the pager idiomatic with the rest of the
    /// rdna-compute API and lets us free via `gpu.free_tensor`.
    tensor: GpuTensor,
    /// Cached byte length so eviction can update `vram_used_bytes` cheaply.
    bytes: u64,
}

/// A pinned-host (page-locked) shadow copy of a paged weight. Backs the
/// v0.3 host tier — when a `WeightId` has a `HostResident`, promoting
/// it to VRAM is a single PCIe-bandwidth memcpy rather than an NVMe
/// read. Stores an offset into `WeightPager::host_arena` rather than
/// owning its own pinned buffer; the arena is one big `hipHostMalloc`
/// shared across all weights, so per-entry residency is just a 16-byte
/// record. Page-locking ~9 GB takes ~300 ms once at warmup vs. ~64 s
/// across 192 small allocs.
struct HostResident {
    /// Byte offset into the pager's `host_arena`.
    offset: usize,
    /// On-disk length of this weight (equal to the pread size). Stored
    /// alongside `offset` so the host→device memcpy can size the
    /// transfer without re-consulting the catalog.
    bytes: u64,
}

impl WeightPager {
    pub fn new(transport: Box<dyn Transport>, config: PagerConfig) -> Self {
        Self {
            resident: HashMap::new(),
            lru: VecDeque::new(),
            vram_used_bytes: 0,
            host_resident: HashMap::new(),
            host_arena: None,
            host_lru: VecDeque::new(),
            host_used_bytes: 0,
            transfer_stream: None,
            transfer_events: Vec::new(),
            catalog: HashMap::new(),
            transport,
            config,
        }
    }

    /// Convenience: open `hfq_path` with the v0.1 pread+H2D transport.
    /// Equivalent to constructing a [`PreadH2DTransport`] manually and passing
    /// it to [`Self::new`].
    pub fn with_pread_transport(hfq_path: &Path, config: PagerConfig) -> std::io::Result<Self> {
        let transport = PreadH2DTransport::open(hfq_path)?;
        Ok(Self::new(Box::new(transport), config))
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

    /// Ensure `id` is in VRAM. Synchronous.
    ///
    /// Three paths, in priority order:
    /// 1. **VRAM hit** — already resident → touch LRU, done.
    /// 2. **Host hit** (v0.3) — pinned host shadow exists → memcpy
    ///    host→device at full PCIe bandwidth (~25 GB/s), no NVMe touch.
    /// 3. **Cold** — read from disk. When `host_budget_bytes > 0`,
    ///    routes the cold load through a freshly-allocated pinned host
    ///    buffer and stashes it in the host tier (so the next eviction
    ///    + re-promotion is a fast PCIe copy). When the host budget is
    ///    `0`, falls back to the v0.2 path (transport.fetch via
    ///    unpinned staging — same semantics as before).
    ///
    /// Errors: `NotRegistered` if `id` was never `register`'d (loader
    /// bug). `BudgetExhausted` if eviction couldn't free enough room.
    pub fn ensure_resident(
        &mut self,
        id: WeightId,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        if self.resident.contains_key(&id) {
            self.touch_lru(id);
            return Ok(());
        }
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        let need = range.len as u64;
        // Evict from VRAM tier if cold-loading would exceed budget.
        // u64::MAX = unlimited (skip the LRU walk).
        if self.config.vram_budget_bytes != u64::MAX
            && self.vram_used_bytes.saturating_add(need) > self.config.vram_budget_bytes
        {
            self.evict_lru_until(need, gpu)?;
        }

        let host_hit = self.host_resident.contains_key(&id)
            || self.try_cold_load_into_host(id, range, gpu)?;
        let tensor = if host_hit {
            let t = {
                let h = self.host_resident.get(&id).unwrap();
                let arena = self.host_arena.as_ref().expect(
                    "host_resident populated without host_arena — pager invariant",
                );
                gpu.upload_raw(&arena[h.offset..h.offset + h.bytes as usize], &[range.len])?
            };
            self.touch_host_lru(id);
            if self.config.trace {
                eprintln!(
                    "[weight_pager] promote-host->vram {id:?} ({} bytes)",
                    range.len
                );
            }
            t
        } else {
            // Host tier disabled or arena full — v0.2 path.
            let (t, _handle) = self.transport.fetch(range.offset, range.len, gpu)?;
            if self.config.trace {
                eprintln!(
                    "[weight_pager] cold-load (no host) {id:?} ({} bytes)",
                    range.len
                );
            }
            t
        };
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(need);
        self.resident.insert(id, Resident { tensor, bytes: need });
        self.lru.push_back(id);
        Ok(())
    }

    /// Populate the pinned-host tier for `id` without touching the GPU.
    /// Used at warmup to pre-fill the host cache so first-touch
    /// promotions to VRAM are PCIe-bound, not NVMe-bound. No-op if the
    /// host tier is disabled (`host_budget_bytes == 0`) or `id` is
    /// already host-resident.
    pub fn prefetch_to_host(
        &mut self,
        id: WeightId,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        if self.config.host_budget_bytes == 0 {
            return Ok(()); // tier disabled
        }
        if self.host_resident.contains_key(&id) {
            self.touch_host_lru(id);
            return Ok(());
        }
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        self.try_cold_load_into_host(id, range, gpu)?;
        Ok(())
    }

    /// Allocate the host arena (lazy, on first prefetch). Sized to
    /// `config.host_budget_bytes`. Plain heap memory — see the field
    /// comment for why we don't pin in v0.3-α.
    fn ensure_host_arena(&mut self, _gpu: &mut Gpu) -> Result<(), WeightPagerError> {
        if self.host_arena.is_some() {
            return Ok(());
        }
        let budget = self.config.host_budget_bytes as usize;
        if budget == 0 {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let buf = vec![0u8; budget];
        if self.config.trace {
            eprintln!(
                "[weight_pager] host arena: allocated {} MB unpinned in {:.2}s",
                budget / (1024 * 1024),
                t0.elapsed().as_secs_f32(),
            );
        }
        self.host_arena = Some(buf);
        Ok(())
    }

    /// **Opportunistic** host-tier population: read `range` from disk
    /// into the next free slot in the host arena and record residency.
    /// Returns `Ok(true)` if the weight got cached, `Ok(false)` if the
    /// arena was full (caller falls back to disk on every page-in for
    /// this weight). Pure no-op when the host tier is disabled.
    ///
    /// "Opportunistic" rather than "must-fit" because the bump arena
    /// has no eviction in v0.3-α — once full, new weights bypass the
    /// host tier entirely. Lets the user set `host_budget_bytes` below
    /// the catalog total without crashing the forward pass.
    fn try_cold_load_into_host(
        &mut self,
        id: WeightId,
        range: ByteRange,
        gpu: &mut Gpu,
    ) -> Result<bool, WeightPagerError> {
        if self.config.host_budget_bytes == 0 {
            return Ok(false);
        }
        let need = range.len as u64;
        if self.host_used_bytes.saturating_add(need) > self.config.host_budget_bytes {
            return Ok(false); // arena full — caller falls back to v0.2 path
        }
        self.ensure_host_arena(gpu)?;
        let offset = self.host_used_bytes as usize;
        {
            let arena = self.host_arena.as_mut().unwrap();
            self.transport.read_to_host(
                range.offset,
                range.len,
                &mut arena[offset..offset + range.len],
            )?;
        }
        self.host_used_bytes = self.host_used_bytes.saturating_add(need);
        self.host_resident.insert(id, HostResident { offset, bytes: need });
        self.host_lru.push_back(id);
        if self.config.trace {
            eprintln!(
                "[weight_pager] host-load {id:?} ({} bytes) @ off {} — {}MB used / {} entries",
                range.len,
                offset,
                self.host_used_bytes / (1024 * 1024),
                self.host_resident.len()
            );
        }
        Ok(true)
    }

    /// Pre-populate the host tier with as many registered weights as
    /// fit in `host_budget_bytes`, in catalog iteration order. Used by
    /// the loader as a one-shot warmup so first-touch promotions are
    /// PCIe-bound (in-RAM) rather than NVMe-bound (on-disk) — for the
    /// portion that fits.
    ///
    /// **Opportunistic**: if the budget is smaller than the catalog
    /// total, the first N weights get cached and the rest fall through
    /// to the v0.2 disk path on every page-in. Caller can also leave
    /// warmup off and let the host tier fill lazily during forward
    /// (same end state on the second pass).
    pub fn prefetch_all_to_host(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        if self.config.host_budget_bytes == 0 {
            return Ok(());
        }
        let ids: Vec<WeightId> = self.catalog.keys().copied().collect();
        for id in ids {
            // try_cold_load_into_host returns Ok(false) when the arena
            // is full — keep iterating so trace stays useful; the
            // remainder simply won't be host-cached.
            self.prefetch_to_host(id, gpu)?;
        }
        Ok(())
    }

    fn touch_host_lru(&mut self, id: WeightId) {
        if let Some(pos) = self.host_lru.iter().position(|x| *x == id) {
            self.host_lru.remove(pos);
            self.host_lru.push_back(id);
        }
    }

    /// Host-tier eviction stub. v0.3-α only supports the
    /// `host_budget_bytes ≥ total paged weights` regime (mode B —
    /// permanent host residency); the bump-arena layout doesn't
    /// support partial reclamation. Returns `BudgetExhausted` when
    /// callers need more room than the arena was sized for. v0.3-β
    /// will introduce a slab/free-list scheme to enable real eviction.
    pub fn evict_host_lru_until(
        &mut self,
        need_bytes: u64,
        _gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        Err(WeightPagerError::BudgetExhausted {
            need_bytes,
            in_use: self.host_used_bytes,
            budget: self.config.host_budget_bytes,
        })
    }

    /// Number of weights currently held in the host tier.
    pub fn host_resident_count(&self) -> usize {
        self.host_resident.len()
    }

    /// Bytes currently held in the host tier.
    pub fn host_used_bytes(&self) -> u64 {
        self.host_used_bytes
    }

    /// **Async** fill: enqueue an H2D copy for `id` into `dst` on the
    /// pager's dedicated transfer stream. Returns an event index that
    /// the caller passes to [`wait_transfer`] before reading `dst`
    /// (e.g. before launching a kernel that consumes it).
    ///
    /// Source is the host tier (v0.3-α arena). When the weight isn't
    /// host-resident, this falls back to the synchronous path —
    /// `transport.read_to_host` is blocking pread, so there's no
    /// async benefit until the weight is in the arena. v0.3-β-α
    /// scope: dense paged forward warms host tier opportunistically
    /// during prefetch_all_to_host, so steady-state is "always async."
    ///
    /// Stream + scratch ordering is the caller's responsibility: this
    /// only guarantees `dst` won't be read by the transfer past
    /// completion of the returned event.
    pub fn fill_into_async(
        &mut self,
        id: WeightId,
        dst: &GpuTensor,
        gpu: &mut Gpu,
    ) -> Result<TransferEventIdx, WeightPagerError> {
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        // Cold-load into host arena if missing (opportunistic — sync
        // pread). If the arena is full, this returns false and we
        // fall back to the v0.2 sync path so the caller doesn't see
        // a partial async semantics.
        let host_hit = self.host_resident.contains_key(&id)
            || self.try_cold_load_into_host(id, range, gpu)?;
        if !host_hit {
            // No host shadow available — sync transport path. The
            // caller still gets an event back so the consumer wait
            // pattern is uniform; we record an event on the default
            // stream after the sync copy.
            self.transport.fill_into(range.offset, range.len, dst, gpu)?;
            return self.record_transfer_on_default_stream(gpu);
        }
        // Host hit: async memcpy on transfer stream + event record.
        self.ensure_transfer_stream(gpu)?;
        let evt = gpu.hip.event_create()?;
        {
            let stream = self.transfer_stream.as_ref().unwrap();
            let h = self.host_resident.get(&id).unwrap();
            let arena = self.host_arena.as_ref().expect(
                "host_resident populated without host_arena — pager invariant",
            );
            gpu.hip.memcpy_htod_async(
                &dst.buf,
                &arena[h.offset..h.offset + h.bytes as usize],
                stream,
            )?;
            gpu.hip.event_record(&evt, Some(stream))?;
        }
        self.touch_host_lru(id);
        let idx = self.transfer_events.len();
        self.transfer_events.push(evt);
        Ok(TransferEventIdx(idx))
    }

    /// Block until the transfer for `idx` completes. Safe to call
    /// after the corresponding `fill_into_async` returns. The event
    /// is consumed (destroyed) — calling `wait_transfer` twice with
    /// the same idx is a usage bug. v0.3-β-α uses one-shot events
    /// without slot reuse for simplicity; v0.3-β-β can pool them if
    /// the per-token event-create overhead becomes visible.
    pub fn wait_transfer(
        &mut self,
        idx: TransferEventIdx,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        let evt = std::mem::replace(
            self.transfer_events
                .get_mut(idx.0)
                .ok_or_else(|| WeightPagerError::Hip(hip_bridge::HipError::new(
                    0,
                    &format!("wait_transfer: event idx {} out of range", idx.0),
                )))?,
            // Replace with a sentinel; we destroy the real event below.
            // event_create on a fresh event is cheap enough at ~μs that
            // a sentinel is acceptable. Alternative: Vec<Option<Event>>.
            gpu.hip.event_create()?,
        );
        gpu.hip.event_synchronize(&evt)?;
        let _ = gpu.hip.event_destroy(evt);
        Ok(())
    }

    fn ensure_transfer_stream(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        if self.transfer_stream.is_none() {
            self.transfer_stream = Some(gpu.hip.stream_create()?);
        }
        Ok(())
    }

    /// Record an event after a synchronous transfer so the caller can
    /// uniformly wait. Used when async fill fell back to sync path
    /// (e.g. host arena was full). The event records on the default
    /// stream so it fires after the prior synchronous copy completes.
    fn record_transfer_on_default_stream(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<TransferEventIdx, WeightPagerError> {
        let evt = gpu.hip.event_create()?;
        // Record on default stream (None) so the event fires once any
        // already-issued default-stream work — including the sync copy
        // we just did — completes.
        gpu.hip.event_record(&evt, None)?;
        let idx = self.transfer_events.len();
        self.transfer_events.push(evt);
        Ok(TransferEventIdx(idx))
    }

    /// Patch the device-side `expert_*_ptrs` indirection table so the indexed
    /// MoE GEMV kernels read the currently-resident buffer pointers for the
    /// active experts in `top_indices` for `layer`.
    ///
    /// The ptr tables are laid out as `[num_experts × u64]` (8-byte device
    /// pointers per expert slot). For each `idx` in `top_indices`, we write
    /// the GPU pointer of that expert's resident gate_up buffer into
    /// `gate_up_ptrs.buf[idx * 8 .. idx * 8 + 8]`, same for down_ptrs.
    ///
    /// Caller must have already called `ensure_resident` for both
    /// `WeightId::Expert{layer, expert: idx, role: GateUp}` and
    /// `WeightId::Expert{layer, expert: idx, role: Down}` for every idx in
    /// `top_indices` — this method asserts that and panics on miss (loader bug).
    pub fn patch_expert_ptr_table(
        &self,
        layer: u16,
        top_indices: &[u16],
        gate_up_ptrs: &GpuTensor,
        down_ptrs: &GpuTensor,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        for &idx in top_indices {
            let gate_up_id = WeightId::Expert { layer, expert: idx, role: ExpertRole::GateUp };
            let down_id = WeightId::Expert { layer, expert: idx, role: ExpertRole::Down };
            let gate_up_tensor = self
                .resident
                .get(&gate_up_id)
                .unwrap_or_else(|| panic!("patch_expert_ptr_table: {gate_up_id:?} not resident"));
            let down_tensor = self
                .resident
                .get(&down_id)
                .unwrap_or_else(|| panic!("patch_expert_ptr_table: {down_id:?} not resident"));
            // u64 pointer values, written into the device table at expert
            // idx's slot. The `as u64` cast is correct because HIP device
            // pointers are 64-bit values on every supported arch (gfx10+,
            // gfx11, gfx12) and the host process is 64-bit; the device-
            // side kernel reads these slots as `unsigned long` and
            // dereferences them as device addresses, so wire format is
            // identical to the host pointer's bit pattern. A non-null
            // assertion catches the rare path where ensure_resident
            // succeeded but the underlying alloc returned a sentinel.
            debug_assert!(!gate_up_tensor.tensor.buf.as_ptr().is_null(),
                "patch_expert_ptr_table: gate_up tensor has null device ptr for {gate_up_id:?}");
            debug_assert!(!down_tensor.tensor.buf.as_ptr().is_null(),
                "patch_expert_ptr_table: down tensor has null device ptr for {down_id:?}");
            let gate_up_ptr = gate_up_tensor.tensor.buf.as_ptr() as u64;
            let down_ptr = down_tensor.tensor.buf.as_ptr() as u64;
            let offset = (idx as usize) * 8;
            gpu.hip.memcpy_htod_offset(&gate_up_ptrs.buf, offset, &gate_up_ptr.to_le_bytes())?;
            gpu.hip.memcpy_htod_offset(&down_ptrs.buf, offset, &down_ptr.to_le_bytes())?;
        }
        Ok(())
    }

    /// Evict residents from the LRU front (least-recently-used) until at
    /// least `need_bytes` would fit under the budget. Returns
    /// [`WeightPagerError::BudgetExhausted`] if nothing more can be evicted
    /// but space is still insufficient.
    ///
    /// Frees evicted tensors via `gpu.free_tensor` — the underlying VRAM
    /// returns to the rdna-compute allocator pool, available for the next
    /// `transport.fetch`.
    pub fn evict_lru_until(
        &mut self,
        need_bytes: u64,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        let budget = self.config.vram_budget_bytes;
        // How much we need to free so that vram_used + need <= budget.
        let target_used = budget.saturating_sub(need_bytes);
        while self.vram_used_bytes > target_used {
            let id = self
                .lru
                .pop_front()
                .ok_or(WeightPagerError::BudgetExhausted {
                    need_bytes,
                    in_use: self.vram_used_bytes,
                    budget,
                })?;
            // Drift guard: the LRU should always contain the same set of
            // ids as `resident`. If they diverge (e.g. a future caller
            // forgets to update both maps in lockstep), we'd silently
            // exit this loop early via BudgetExhausted instead of
            // hitting the actual bug. Trap drift in debug builds; the
            // release path stays the same `if let Some(_)` defensive
            // continue (worst case: we evict more than necessary).
            debug_assert!(
                self.resident.contains_key(&id),
                "weight_pager invariant: lru contains {id:?} but resident does not — \
                 drift between LRU queue and residency map"
            );
            if let Some(r) = self.resident.remove(&id) {
                self.vram_used_bytes = self.vram_used_bytes.saturating_sub(r.bytes);
                let _ = gpu.free_tensor(r.tensor);
                if self.config.trace {
                    eprintln!(
                        "[weight_pager] evict {id:?} ({} bytes) — {} resident, {} used",
                        r.bytes,
                        self.resident.len(),
                        self.vram_used_bytes
                    );
                }
            }
        }
        Ok(())
    }

    /// Free all resident tensors back to the GPU pool. Also frees the
    /// pinned-host arena. Called on model teardown so VRAM and locked
    /// RAM both return to the system. After this, the pager is
    /// effectively reset (catalog stays, residency maps are empty).
    pub fn free_all(&mut self, gpu: &mut Gpu) {
        for (_id, r) in self.resident.drain() {
            let _ = gpu.free_tensor(r.tensor);
        }
        self.lru.clear();
        self.vram_used_bytes = 0;
        self.host_resident.clear();
        self.host_lru.clear();
        self.host_used_bytes = 0;
        // Vec<u8> arena drops naturally via take().
        let _ = self.host_arena.take();
        // Tear down the transfer stream + drain any lingering events.
        // Sync first (waits for any in-flight copy) so destroying the
        // event objects doesn't yank state out from under the GPU.
        for evt in self.transfer_events.drain(..) {
            let _ = gpu.hip.event_destroy(evt);
        }
        if let Some(stream) = self.transfer_stream.take() {
            let _ = gpu.hip.stream_synchronize(&stream);
            let _ = gpu.hip.stream_destroy(stream);
        }
    }

    /// Get the resident tensor for `id`. Returns `None` if not resident
    /// (caller should `ensure_resident` first). Does not affect LRU.
    pub fn get(&self, id: WeightId) -> Option<&GpuTensor> {
        self.resident.get(&id).map(|r| &r.tensor)
    }

    /// Fill an existing device buffer `dst` with the bytes for `id`. Used
    /// by the scratch-buffer paging pattern (dense FFN paging) where one
    /// fixed-size scratch tensor is reused across layers — the pager
    /// rewrites the underlying memory each time the active layer changes
    /// rather than allocating fresh GpuTensors.
    ///
    /// **Host-tier-aware** (v0.3): when a pinned host shadow exists for
    /// `id`, copies host→device at PCIe bandwidth and skips the disk
    /// read entirely. When the host budget is set but `id` isn't yet
    /// shadowed, the cold load goes through a freshly-allocated pinned
    /// buffer that gets stashed for next time. With `host_budget == 0`,
    /// falls back to the v0.2 path (`transport.fill_into` via unpinned
    /// staging).
    ///
    /// Doesn't update the VRAM residency map (no entry is created — the
    /// scratch buffer is caller-owned). Does touch host LRU on hit.
    pub fn fill_into(
        &mut self,
        id: WeightId,
        dst: &GpuTensor,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        let host_hit = self.host_resident.contains_key(&id)
            || self.try_cold_load_into_host(id, range, gpu)?;
        if host_hit {
            {
                let h = self.host_resident.get(&id).unwrap();
                let arena = self.host_arena.as_ref().expect(
                    "host_resident populated without host_arena — pager invariant",
                );
                gpu.hip.memcpy_htod(
                    &dst.buf,
                    &arena[h.offset..h.offset + h.bytes as usize],
                )?;
            }
            self.touch_host_lru(id);
            return Ok(());
        }
        // Host tier disabled or arena full — v0.2 path.
        self.transport.fill_into(range.offset, range.len, dst, gpu)?;
        Ok(())
    }

    /// Insert an already-resident weight. Used by the loader for
    /// always-resident weights (token embeds, norms, the router itself in
    /// v0.1) — they live in VRAM from startup but the pager tracks them so
    /// they're visible to `get()` and accounted in `vram_used_bytes`.
    pub fn insert_resident(&mut self, id: WeightId, tensor: GpuTensor, bytes: u64) {
        if let Some(prev) = self.resident.remove(&id) {
            self.vram_used_bytes = self.vram_used_bytes.saturating_sub(prev.bytes);
            self.lru.retain(|x| *x != id);
        }
        self.resident.insert(id, Resident { tensor, bytes });
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

    /// Bytes currently held resident. Cheap (cached, not a sum).
    pub fn vram_used_bytes(&self) -> u64 {
        self.vram_used_bytes
    }

    /// Number of currently-resident weights.
    pub fn resident_count(&self) -> usize {
        self.resident.len()
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
    /// Eviction couldn't free enough room — budget too small for the
    /// requested weight. User needs to raise `vram_budget_bytes` or
    /// reduce the working set somehow.
    BudgetExhausted {
        need_bytes: u64,
        in_use: u64,
        budget: u64,
    },
    /// Stub for paths still being filled in.
    Unimplemented(&'static str),
}

impl std::fmt::Display for WeightPagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRegistered(id) => write!(f, "weight not registered: {id:?}"),
            Self::Hip(e) => write!(f, "hip error: {e}"),
            Self::BudgetExhausted { need_bytes, in_use, budget } => write!(
                f,
                "weight pager: cannot evict to fit {need_bytes} bytes \
                 (in_use={in_use}, budget={budget}); raise vram_budget_bytes \
                 or reduce paged working set"
            ),
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
    use std::io::Write;

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

    /// Write some bytes to a temp file and verify `PreadH2DTransport::open`
    /// can read arbitrary ranges via the staging buffer. The actual upload
    /// to GPU is exercised in integration tests; here we directly call the
    /// pread helper to keep the unit test device-free.
    #[test]
    fn pread_transport_reads_range() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-test-{}.bin", std::process::id()));
        let payload: Vec<u8> = (0..1024u32).flat_map(|i| (i as u8).to_le_bytes()).collect();
        std::fs::File::create(&path).unwrap().write_all(&payload).unwrap();

        let mut t = PreadH2DTransport::open(&path).unwrap();
        // Read [256..768) — should match payload[256..768].
        t.pread_into_staging(256, 512).unwrap();
        assert_eq!(&t.staging[..512], &payload[256..768]);
        // Read a smaller range; staging must cover it.
        t.pread_into_staging(0, 16).unwrap();
        assert_eq!(&t.staging[..16], &payload[..16]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pager_starts_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-empty-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        assert_eq!(pager.registered_count(), 0);
        assert_eq!(pager.vram_used_bytes(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn register_then_get_returns_none_until_resident() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-reg-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let mut pager =
            WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        let id = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::GateUp };
        pager.register(id, ByteRange { offset: 0, len: 1 });
        assert_eq!(pager.registered_count(), 1);
        // Catalog hit, not yet resident → get returns None.
        assert!(!pager.is_resident(id));
        assert!(pager.get(id).is_none());
        // ensure_resident requires a real Gpu — exercised in integration tests.
        let _ = std::fs::remove_file(&path);
    }
}
