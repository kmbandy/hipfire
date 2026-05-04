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

    /// **Predictive readahead hint** (v0.3-ε). Caller is about to need
    /// these byte ranges; transport may pre-warm internal caches or
    /// kick off OS-level readahead so subsequent `read_to_host` /
    /// `fill_into` calls hit warm state. Best-effort — kernel is free
    /// to ignore.
    ///
    /// Designed to be called from the forward path while the GPU is
    /// busy on the current layer's compute, naming the next K layers'
    /// weight ranges. The kernel's NVMe driver can issue many parallel
    /// reads in the background while our process is otherwise occupied,
    /// extracting the drive's full QD-N bandwidth without us having to
    /// build our own thread pool. Default impl is a no-op so transports
    /// that don't benefit (in-memory mocks, future P2P-DMA paths) just
    /// ignore it.
    fn advise_prefetch(&self, ranges: &[ByteRange]) {
        let _ = ranges;
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

    /// `posix_fadvise(WILLNEED)` per range — tells the kernel to pull
    /// these byte ranges into page cache via background readahead.
    /// Subsequent `pread` against the same ranges then hits warm cache
    /// (memcpy speed, ~10 GB/s) instead of going to the NVMe (~500 MB/s
    /// QD=1 sequential ceiling on a healthy SN850X).
    ///
    /// Note: the file-wide `POSIX_FADV_RANDOM` set in `open` disables
    /// the *automatic* sequential-readahead heuristic; explicit
    /// `WILLNEED` hints are still honored independently. So we get
    /// per-tensor readahead exactly where we ask for it without the
    /// kernel speculating across unrelated tensors.
    fn advise_prefetch(&self, ranges: &[ByteRange]) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self.file.as_raw_fd();
            for r in ranges {
                // Best-effort: ignore the return code. EBADF can't
                // happen (fd is owned), EINVAL would mean an absurd
                // offset/len combo we'd hit elsewhere first.
                unsafe {
                    libc::posix_fadvise(
                        fd,
                        r.offset as libc::off_t,
                        r.len as libc::off_t,
                        libc::POSIX_FADV_WILLNEED,
                    );
                }
            }
        }
        let _ = ranges;
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
    /// Carved into per-size slots by [`Self::host_pools`] +
    /// [`Self::host_high_water`]; the arena buffer itself never
    /// reallocates after `ensure_host_arena`.
    ///
    /// Tries pinned (`hipHostMalloc` with the AMD `Coherent` flag,
    /// which uses IOMMU mapping rather than `mlock` and bypasses
    /// `RLIMIT_MEMLOCK`) first; falls back to a pageable `Vec<u8>`
    /// when ROCm refuses pinning. **Pinned is required for actual
    /// async overlap** — `hipMemcpyAsync` from pageable host memory
    /// is documented-sync per AMD, so the v0.3-β async forward path
    /// only delivers parallelism with a pinned arena.
    host_arena: Option<HostArena>,
    /// LRU queue for the host tier. Front = least-recently-used eviction
    /// victim; back = most-recently-touched. Consulted by
    /// [`Self::evict_one_host_lru`] when an allocation can neither be
    /// satisfied from a same-size pool slot nor carved from the tail.
    host_lru: VecDeque<WeightId>,
    /// Bytes currently held by live `host_resident` entries. **Logical
    /// occupancy**, not the arena cursor — freed pool slots are
    /// subtracted on eviction, so this is the honest "how much host
    /// RAM is in active use" number returned by [`Self::host_used_bytes`].
    host_used_bytes: u64,
    /// Size-class free lists: `byte_size → free_offsets`. When a host
    /// slot is evicted, its arena offset goes here under the slot's
    /// byte size; the next allocation of that exact size pops from the
    /// list instead of carving fresh tail space. Hipfire's MoE-dominated
    /// paged workload is uniform-shape per (layer, role), so same-size
    /// reuse is the dominant path.
    host_pools: HashMap<usize, Vec<usize>>,
    /// Bump cursor for first-time slot creation. Grows monotonically
    /// (never recedes — pool slots handle reuse). When
    /// `host_high_water + need > arena_cap` and there's no matching
    /// pool entry, the allocator evicts an LRU slot and retries.
    host_high_water: usize,
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

/// Big shared arena for the host tier. Tagged so callers can tell
/// pinned (DMA-fast, async-capable) from pageable (sync-only) and
/// switch the forward-path overlap loop on/off accordingly.
enum HostArena {
    /// `hipHostMalloc`-backed pinned region. `hipMemcpyAsync` from
    /// here is genuinely asynchronous and runs at ~25 GB/s on PCIe
    /// 4.0; required for the v0.3-β compute/transfer overlap loop.
    Pinned(hip_bridge::HostBuffer),
    /// Plain heap-allocated bytes. Works as a disk-skip cache but
    /// `hipMemcpyAsync` is documented-sync from here — the forward
    /// path falls back to the v0.3-α single-slot sync fill in this
    /// mode rather than paying the event-create overhead with no
    /// overlap to show for it.
    Pageable(Vec<u8>),
}

impl HostArena {
    fn as_slice(&self) -> &[u8] {
        match self {
            HostArena::Pinned(b) => b.as_slice(),
            HostArena::Pageable(v) => v.as_slice(),
        }
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            HostArena::Pinned(b) => b.as_mut_slice(),
            HostArena::Pageable(v) => v.as_mut_slice(),
        }
    }
    fn is_pinned(&self) -> bool {
        matches!(self, HostArena::Pinned(_))
    }
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
            host_pools: HashMap::new(),
            host_high_water: 0,
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
                gpu.upload_raw(&arena.as_slice()[h.offset..h.offset + h.bytes as usize], &[range.len])?
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

    /// **Async** ensure_resident: like [`ensure_resident`] but issues
    /// the host→device copy on the pager's transfer stream. Returns:
    ///
    /// - `None` when the weight is already VRAM-resident (LRU touched,
    ///   no work scheduled). Critical for warm-cache workloads — the
    ///   async-batch MoE forward can hit this 99%+ of the time at
    ///   steady state, and the `None` short-circuit means it adds zero
    ///   event overhead vs the sync path.
    /// - `Some(idx)` when an async H2D was issued (cold + host-hit).
    ///   Caller must `wait_transfer` before launching a kernel that
    ///   reads the weight.
    /// - `None` when the weight was sync-fetched from disk (cold +
    ///   host-miss). By the time we return, the bytes are in VRAM and
    ///   the residency map is updated; no wait needed.
    pub fn ensure_resident_async(
        &mut self,
        id: WeightId,
        gpu: &mut Gpu,
    ) -> Result<Option<TransferEventIdx>, WeightPagerError> {
        if self.resident.contains_key(&id) {
            self.touch_lru(id);
            return Ok(None);
        }
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        let need = range.len as u64;
        if self.config.vram_budget_bytes != u64::MAX
            && self.vram_used_bytes.saturating_add(need) > self.config.vram_budget_bytes
        {
            self.evict_lru_until(need, gpu)?;
        }

        let host_hit = self.host_resident.contains_key(&id)
            || self.try_cold_load_into_host(id, range, gpu)?;
        if host_hit {
            // Async H2D from pinned arena → fresh device buffer.
            self.ensure_transfer_stream(gpu)?;
            let device_buf = gpu.hip.malloc(range.len)?;
            let evt = gpu.hip.event_create()?;
            {
                let stream = self.transfer_stream.as_ref().unwrap();
                let h = self.host_resident.get(&id).unwrap();
                let arena = self.host_arena.as_ref().expect(
                    "host_resident populated without host_arena — pager invariant",
                );
                gpu.hip.memcpy_htod_async(
                    &device_buf,
                    &arena.as_slice()[h.offset..h.offset + h.bytes as usize],
                    stream,
                )?;
                gpu.hip.event_record(&evt, Some(stream))?;
            }
            self.touch_host_lru(id);
            let tensor = GpuTensor {
                buf: device_buf,
                shape: vec![range.len],
                dtype: rdna_compute::DType::Raw,
            };
            self.vram_used_bytes = self.vram_used_bytes.saturating_add(need);
            self.resident.insert(id, Resident { tensor, bytes: need });
            self.lru.push_back(id);
            let idx = self.transfer_events.len();
            self.transfer_events.push(evt);
            return Ok(Some(TransferEventIdx(idx)));
        }

        // Cold + host-miss — fall back to sync fetch. By return-time
        // the bytes are already in VRAM; no event needed.
        let (tensor, _h) = self.transport.fetch(range.offset, range.len, gpu)?;
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(need);
        self.resident.insert(id, Resident { tensor, bytes: need });
        self.lru.push_back(id);
        if self.config.trace {
            eprintln!(
                "[weight_pager] cold-load (async fallback) {id:?} ({} bytes)",
                range.len
            );
        }
        Ok(None)
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

    /// Allocate the host arena (lazy, on first prefetch). Tries pinned
    /// (`hipHostMalloc` with the AMD `Coherent` flag) first; falls
    /// back to plain `Vec<u8>` on any HIP error so the host tier keeps
    /// working even when pinning is denied.
    ///
    /// **Pre-flight clamp**: AMD's Coherent allocation succeeds even
    /// when there isn't actually that much RAM available — the
    /// IOMMU-mapped pages are eligible for swap when the system is
    /// pressured. So a request that's too big for the box's actual
    /// headroom lands silently, then the moment we fault each page
    /// the OS starts swapping out everything else and the whole
    /// process grinds to a halt (~25 sec per "compute" step on a
    /// 16 GB box). We read `MemAvailable` from `/proc/meminfo` and
    /// clamp the request to leave a 2 GB safety margin for the
    /// kernel + page cache + other processes.
    ///
    /// **Opt-in `mlock`** (`HIPFIRE_PIN_HOST=1`): even with the
    /// pre-flight clamp, Coherent pages stay swap-eligible because
    /// the IOMMU mapping doesn't page-lock them. Under page-cache
    /// pressure (e.g. a multi-GB model file `mmap`'d alongside our
    /// arena) the kernel will silently swap out arena pages it
    /// thinks are idle; subsequent `memcpy_htod_async` then page-
    /// faults from disk-backed swap. Setting `HIPFIRE_PIN_HOST=1`
    /// calls `mlock` on the arena right after allocation, making
    /// the pages truly unswappable. Subject to `RLIMIT_MEMLOCK` —
    /// if `mlock` returns `EPERM`/`ENOMEM` we log a clear error and
    /// continue without pinning (arena still works, just swap-eligible).
    /// Remediation: `ulimit -l unlimited` for the shell, or set
    /// `* hard memlock unlimited` in `/etc/security/limits.conf`.
    fn ensure_host_arena(&mut self, gpu: &mut Gpu) -> Result<(), WeightPagerError> {
        if self.host_arena.is_some() {
            return Ok(());
        }
        let mut budget = self.config.host_budget_bytes as usize;
        if budget == 0 {
            return Ok(());
        }
        // Pre-flight: clamp against actual system memory availability.
        if let Some(avail) = read_mem_available() {
            const SAFETY_MARGIN: usize = 2 * 1024 * 1024 * 1024; // 2 GB
            let safe_cap = avail.saturating_sub(SAFETY_MARGIN);
            if budget > safe_cap {
                eprintln!(
                    "[weight_pager] host arena: requested {} MB but only {} MB available \
                     (after 2 GB safety margin from /proc/meminfo MemAvailable={} MB) — \
                     clamping to {} MB. Allocating beyond available RAM forces swap \
                     thrashing on AMD Coherent (IOMMU-backed pages stay swappable). \
                     Close other processes or raise system RAM for the full requested \
                     budget.",
                    budget / (1024 * 1024),
                    safe_cap / (1024 * 1024),
                    avail / (1024 * 1024),
                    safe_cap / (1024 * 1024),
                );
                budget = safe_cap;
                if budget == 0 {
                    return Err(WeightPagerError::BudgetExhausted {
                        need_bytes: self.config.host_budget_bytes,
                        in_use: 0,
                        budget: 0,
                    });
                }
            }
        } else if self.config.trace {
            eprintln!("[weight_pager] could not read /proc/meminfo — skipping pre-flight clamp");
        }
        let t0 = std::time::Instant::now();
        // Try pinned (AMD Coherent — IOMMU-backed, bypasses
        // RLIMIT_MEMLOCK on systems with USM). Fall back to pageable
        // on any HIP error rather than aborting the load.
        const HIP_HOST_MALLOC_COHERENT: u32 = 0x40000000;
        match gpu.hip.host_malloc_with_flags(budget, HIP_HOST_MALLOC_COHERENT) {
            Ok(buf) => {
                eprintln!(
                    "[weight_pager] host arena: allocated {} MB PINNED (Coherent) in {:.2}s — async overlap available",
                    budget / (1024 * 1024),
                    t0.elapsed().as_secs_f32(),
                );
                if std::env::var_os("HIPFIRE_PIN_HOST").is_some() {
                    let t_mlock = std::time::Instant::now();
                    // SAFETY: buf.as_ptr() points at `buf.size()` bytes of
                    // valid Coherent host memory we just allocated. mlock
                    // is read-only against the buffer contents (it walks
                    // page tables and pins them).
                    let rc = unsafe {
                        libc::mlock(buf.as_ptr() as *const libc::c_void, buf.size())
                    };
                    if rc == 0 {
                        eprintln!(
                            "[weight_pager] host arena: mlock'd {} MB in {:.2}s — \
                             arena is now unswappable",
                            buf.size() / (1024 * 1024),
                            t_mlock.elapsed().as_secs_f32(),
                        );
                    } else {
                        let errno = std::io::Error::last_os_error();
                        eprintln!(
                            "[weight_pager] host arena: mlock denied ({errno}) — arena \
                             stays swap-eligible. Remediation: `ulimit -l unlimited` in \
                             this shell, or `* hard memlock unlimited` in \
                             /etc/security/limits.conf (then re-login). Continuing \
                             without pin — under heavy page-cache pressure (e.g. large \
                             mmap'd model file), expect occasional swap-fault stalls."
                        );
                    }
                }
                self.host_arena = Some(HostArena::Pinned(buf));
            }
            Err(e) => {
                eprintln!(
                    "[weight_pager] host arena: pinned alloc denied ({e}); falling back to pageable Vec<u8> — overlap loop will use sync path"
                );
                let v = vec![0u8; budget];
                eprintln!(
                    "[weight_pager] host arena: allocated {} MB pageable in {:.2}s",
                    budget / (1024 * 1024),
                    t0.elapsed().as_secs_f32(),
                );
                self.host_arena = Some(HostArena::Pageable(v));
            }
        }
        // Update the effective budget so try_cold_load_into_host's
        // overflow check uses the clamped value rather than the
        // operator's request.
        self.config.host_budget_bytes = budget as u64;
        Ok(())
    }

    /// `true` once the host arena is allocated AND backed by pinned
    /// memory. The forward path checks this before engaging the v0.3-β
    /// overlap loop — `hipMemcpyAsync` from pageable host is sync, so
    /// running the overlap loop without pinning costs event overhead
    /// for nothing.
    pub fn host_arena_pinned(&self) -> bool {
        matches!(&self.host_arena, Some(a) if a.is_pinned())
    }

    /// Slab-allocated host-tier population: read `range` from disk into
    /// a host arena slot and record residency. Returns `Ok(true)` if
    /// the weight got cached, `Ok(false)` only as the final escape
    /// hatch (host tier disabled, single weight bigger than the entire
    /// budget, or pathological size-mix saturation — see below). Pure
    /// no-op when the host tier is disabled.
    ///
    /// **Slot acquisition order** (v0.3-δ):
    /// 1. **Pool hit** — reuse a free slot of the exact byte size left
    ///    behind by a prior eviction. Dominant path for MoE workloads
    ///    where every (layer, role) shares a size.
    /// 2. **Tail carve** — bump `host_high_water` if the arena tail
    ///    has room. The v0.3-α/β/γ fast path during warmup.
    /// 3. **Evict + retry** — pop an LRU host slot, return its offset
    ///    to its size pool, retry from step 1. Loops until the
    ///    allocation lands or `host_lru` is exhausted.
    ///
    /// **Returns `Ok(false)` only when:**
    /// - `host_budget_bytes == 0` (tier disabled).
    /// - `range.len > host_budget_bytes` (single weight too big — no
    ///   allocator strategy fixes this; raise the budget).
    /// - High water saturated AND no resident slot has size `range.len`,
    ///   so eviction can never produce a usable pool slot. Cross-size
    ///   pathological case; benign for MoE (uniform sizes), can occur
    ///   in heavily mixed dense+MoE workloads. Caller falls back to
    ///   the v0.2 disk path for that one weight rather than crashing.
    fn try_cold_load_into_host(
        &mut self,
        id: WeightId,
        range: ByteRange,
        gpu: &mut Gpu,
    ) -> Result<bool, WeightPagerError> {
        if self.config.host_budget_bytes == 0 {
            return Ok(false);
        }
        // ensure_host_arena clamps the budget to MemAvailable, so do it
        // up front and read the *clamped* cap below — otherwise the
        // allocator could carve past what was actually mapped.
        self.ensure_host_arena(gpu)?;
        let arena_cap = self.config.host_budget_bytes as usize;
        if range.len > arena_cap {
            // Single weight bigger than the entire (clamped) arena. No
            // amount of eviction creates room. Caller falls back to disk.
            if self.config.trace {
                eprintln!(
                    "[weight_pager] host-load skip {id:?}: weight {} bytes > arena cap {} bytes",
                    range.len, arena_cap,
                );
            }
            return Ok(false);
        }

        // One-shot stream sync (lazy, only if we end up evicting). An
        // in-flight memcpy_htod_async could be reading a slot we're
        // about to recycle; sync the transfer stream once before any
        // eviction fires so the recycled offset is safe to overwrite.
        let mut transfer_synced = false;
        let offset = loop {
            // (1) Pool hit — same-size slot left over from an eviction.
            if let Some(off) = self
                .host_pools
                .get_mut(&range.len)
                .and_then(|v| v.pop())
            {
                break off;
            }
            // (2) Tail carve — first-time slot creation.
            if self.host_high_water + range.len <= arena_cap {
                let off = self.host_high_water;
                self.host_high_water += range.len;
                break off;
            }
            // (3) Evict one and retry. Sync once before the first
            // eviction; subsequent iterations see `transfer_synced=true`
            // and skip the redundant wait.
            if !transfer_synced {
                if let Some(stream) = self.transfer_stream.as_ref() {
                    gpu.hip.stream_synchronize(stream)?;
                }
                transfer_synced = true;
            }
            if !self.evict_one_host_lru() {
                // LRU empty: arena is fully carved, no pool slot of
                // this size, and nothing left to evict. Cross-size
                // saturation (e.g. arena full of size-A slots, asking
                // for size-B that has zero residents). Caller falls
                // back to v0.2 disk path for this weight.
                if self.config.trace {
                    eprintln!(
                        "[weight_pager] host-load skip {id:?}: arena saturated with no \
                         size-{} slots and LRU exhausted",
                        range.len,
                    );
                }
                return Ok(false);
            }
        };

        let need = range.len as u64;
        {
            let arena = self.host_arena.as_mut().unwrap();
            self.transport.read_to_host(
                range.offset,
                range.len,
                &mut arena.as_mut_slice()[offset..offset + range.len],
            )?;
        }
        self.host_used_bytes = self.host_used_bytes.saturating_add(need);
        self.host_resident.insert(id, HostResident { offset, bytes: need });
        self.host_lru.push_back(id);
        if self.config.trace {
            eprintln!(
                "[weight_pager] host-load {id:?} ({} bytes) @ off {} — \
                 used {}MB / hi-water {}MB / {} entries",
                range.len,
                offset,
                self.host_used_bytes / (1024 * 1024),
                self.host_high_water / (1024 * 1024),
                self.host_resident.len()
            );
        }
        Ok(true)
    }

    /// Pop the LRU front, free its slot back to the size pool, and
    /// update bookkeeping. Returns `false` if the LRU was empty (caller
    /// has nothing more to evict). Does **not** sync the transfer
    /// stream — caller is responsible for that one-shot wait before
    /// the first eviction in a sequence (see [`Self::try_cold_load_into_host`]
    /// and [`Self::evict_host_lru_until`]).
    fn evict_one_host_lru(&mut self) -> bool {
        let id = match self.host_lru.pop_front() {
            Some(id) => id,
            None => return false,
        };
        let h = self
            .host_resident
            .remove(&id)
            .expect("host_lru/host_resident invariant: lru has id but residency map does not");
        self.host_pools
            .entry(h.bytes as usize)
            .or_default()
            .push(h.offset);
        self.host_used_bytes = self.host_used_bytes.saturating_sub(h.bytes);
        if self.config.trace {
            let pool_free = self
                .host_pools
                .get(&(h.bytes as usize))
                .map(|v| v.len())
                .unwrap_or(0);
            eprintln!(
                "[weight_pager] host-evict {id:?} ({} bytes) → pool[{}]={} free, used={}MB",
                h.bytes,
                h.bytes,
                pool_free,
                self.host_used_bytes / (1024 * 1024),
            );
        }
        true
    }

    /// Pre-populate the host tier with as many registered weights as
    /// fit in `host_budget_bytes`, in catalog iteration order. Used by
    /// the loader as a one-shot warmup so first-touch promotions are
    /// PCIe-bound (in-RAM) rather than NVMe-bound (on-disk) — for the
    /// portion that fits.
    ///
    /// **Stops at budget**: when the next weight would force eviction,
    /// the loop exits early. With v0.3-δ slab eviction enabled, naively
    /// continuing would LRU-thrash the arena — load slot 1, then evict
    /// it to load slot (cap+1), then evict that to load slot (cap+2),
    /// etc. The forward path pulls what it actually needs lazily, so
    /// warmup beyond the budget would just generate wasted NVMe reads
    /// for weights that immediately get displaced.
    pub fn prefetch_all_to_host(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        if self.config.host_budget_bytes == 0 {
            return Ok(());
        }
        let cap = self.config.host_budget_bytes;
        let ids: Vec<WeightId> = self.catalog.keys().copied().collect();
        for id in ids {
            let need = match self.catalog.get(&id) {
                Some(r) => r.len as u64,
                None => continue,
            };
            if self.host_used_bytes.saturating_add(need) > cap {
                if self.config.trace {
                    eprintln!(
                        "[weight_pager] prefetch_all_to_host: budget reached at {} entries, \
                         {} MB used / {} MB cap — remaining {} weights will load lazily",
                        self.host_resident.len(),
                        self.host_used_bytes / (1024 * 1024),
                        cap / (1024 * 1024),
                        self.catalog.len().saturating_sub(self.host_resident.len()),
                    );
                }
                break;
            }
            self.prefetch_to_host(id, gpu)?;
        }
        Ok(())
    }

    /// **Predictive readahead** (v0.3-ε): tell the OS we're about to
    /// need these weights. Maps each [`WeightId`] to its byte range via
    /// the catalog and forwards to [`Transport::advise_prefetch`].
    /// Skips ids that are already host-resident (no disk re-read needed)
    /// and ids missing from the catalog (loader bug — hint is best-effort,
    /// not worth panicking).
    ///
    /// **Intended call site**: from the forward path while the GPU is
    /// busy on the current layer's compute, name the next K layers'
    /// weights. The kernel's NVMe driver issues parallel readaheads in
    /// the background; subsequent cold-loads in `try_cold_load_into_host`
    /// pread from warm page cache instead of the drive, which lifts the
    /// effective transport bandwidth from the QD=1 sequential ceiling
    /// (~500 MB/s on SN850X) toward what a realistic concurrent workload
    /// can extract (3+ GB/s on the same drive).
    ///
    /// Cheap to call (one syscall per range, no I/O on the calling
    /// thread). Safe to over-call — duplicate `WILLNEED` hints are
    /// idempotent at the kernel level.
    pub fn advise_prefetch(&self, ids: &[WeightId]) {
        if ids.is_empty() {
            return;
        }
        let mut ranges: Vec<ByteRange> = Vec::with_capacity(ids.len());
        for id in ids {
            // Skip already-host-resident: the bytes are in our pinned
            // arena, no disk read happens for them. Skip uncatalogued:
            // loader-side bug, not the prefetch path's job to surface.
            if self.host_resident.contains_key(id) {
                continue;
            }
            if let Some(range) = self.catalog.get(id) {
                ranges.push(*range);
            }
        }
        if !ranges.is_empty() {
            self.transport.advise_prefetch(&ranges);
        }
    }

    fn touch_host_lru(&mut self, id: WeightId) {
        if let Some(pos) = self.host_lru.iter().position(|x| *x == id) {
            self.host_lru.remove(pos);
            self.host_lru.push_back(id);
        }
    }

    /// Free host-tier slots until `host_used_bytes + need_bytes`
    /// fits under the configured budget. Pops LRU front entries one
    /// at a time, returning each freed offset to its size pool so
    /// subsequent same-size allocations reuse it.
    ///
    /// **Note**: this is a logical-occupancy guarantee, not a
    /// "you can definitely allocate `need_bytes` of size N right now"
    /// guarantee — same-size pool slots become available immediately,
    /// but cross-size carves still depend on `host_high_water` having
    /// room. [`Self::try_cold_load_into_host`] does its own
    /// allocation-aware eviction loop; this method exists for
    /// callers that want to proactively free room (e.g. a future
    /// predictive-prefetch layer).
    ///
    /// Returns `BudgetExhausted` when the LRU is exhausted and
    /// `host_used_bytes + need_bytes` still exceeds the budget.
    pub fn evict_host_lru_until(
        &mut self,
        need_bytes: u64,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        if self.config.host_budget_bytes == 0 {
            return Ok(());
        }
        let budget = self.config.host_budget_bytes;
        if self.host_used_bytes.saturating_add(need_bytes) <= budget {
            return Ok(());
        }
        // One-shot transfer-stream sync before recycling any slot that
        // an in-flight async H2D might still be reading.
        if let Some(stream) = self.transfer_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        }
        while self.host_used_bytes.saturating_add(need_bytes) > budget {
            if !self.evict_one_host_lru() {
                return Err(WeightPagerError::BudgetExhausted {
                    need_bytes,
                    in_use: self.host_used_bytes,
                    budget,
                });
            }
        }
        Ok(())
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
    /// pager's dedicated transfer stream. Returns:
    ///
    /// - `Some(idx)` when the copy was issued asynchronously (host-hit
    ///   path, real overlap potential). Caller must `wait_transfer(idx)`
    ///   before reading `dst`.
    /// - `None` when the work completed synchronously before this
    ///   function returned (host-miss → blocking transport.fill_into).
    ///   Caller can use `dst` immediately, no wait needed.
    ///
    /// `Option` lets callers cheaply skip event ceremony on the sync
    /// path — event_create + record + sync + destroy is ~tens of
    /// microseconds per call and dominates per-token overhead when
    /// most calls are no-async (e.g. warm-cache MoE inference where
    /// the VRAM-hit fast path exists).
    pub fn fill_into_async(
        &mut self,
        id: WeightId,
        dst: &GpuTensor,
        gpu: &mut Gpu,
    ) -> Result<Option<TransferEventIdx>, WeightPagerError> {
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        // Cold-load into host arena if missing (opportunistic — sync
        // pread). If the arena is full, this returns false and we
        // fall back to the v0.2 sync path.
        let host_hit = self.host_resident.contains_key(&id)
            || self.try_cold_load_into_host(id, range, gpu)?;
        if !host_hit {
            // No host shadow — sync transport path. By the time it
            // returns the data is in `dst`; no event needed.
            self.transport.fill_into(range.offset, range.len, dst, gpu)?;
            return Ok(None);
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
                &arena.as_slice()[h.offset..h.offset + h.bytes as usize],
                stream,
            )?;
            gpu.hip.event_record(&evt, Some(stream))?;
        }
        self.touch_host_lru(id);
        let idx = self.transfer_events.len();
        self.transfer_events.push(evt);
        Ok(Some(TransferEventIdx(idx)))
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
            // Non-blocking is mandatory for actual compute/transfer
            // overlap. The default hipStreamCreate would implicitly
            // serialize against the default stream and we'd see the
            // event-overhead with none of the overlap benefit.
            self.transfer_stream = Some(gpu.hip.stream_create_nonblocking()?);
        }
        Ok(())
    }

    /// Make the pager's transfer stream wait on a compute event before
    /// proceeding with any subsequently-enqueued async transfer. Used
    /// by the forward path's overlap loop to defend against the
    /// 2-slot-reuse hazard: when prefetching for slot S that was last
    /// read by compute layer N-1, the transfer stream must wait on
    /// compute_event[N-1] so it doesn't overwrite bytes still being
    /// consumed by an in-flight kernel. Lazily creates the transfer
    /// stream if not yet allocated.
    pub fn transfer_wait_for_compute(
        &mut self,
        compute_event: &hip_bridge::Event,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        self.ensure_transfer_stream(gpu)?;
        let stream = self.transfer_stream.as_ref().unwrap();
        gpu.hip.stream_wait_event(stream, compute_event)?;
        Ok(())
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
        self.host_pools.clear();
        self.host_high_water = 0;
        // Pinned arena needs hipHostFree; pageable just drops.
        match self.host_arena.take() {
            Some(HostArena::Pinned(buf)) => { let _ = gpu.hip.host_free(buf); }
            Some(HostArena::Pageable(_)) | None => {}
        }
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
                    &arena.as_slice()[h.offset..h.offset + h.bytes as usize],
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

/// Parse `/proc/meminfo` for the `MemAvailable:` field — bytes the
/// kernel believes can be allocated to userspace right now without
/// going to swap. Used by the pager's host-arena pre-flight to clamp
/// outsized budgets before the OS starts thrashing. Returns `None` on
/// non-Linux or any parse error (caller skips the clamp).
fn read_mem_available() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // Format: "MemAvailable:    9258484 kB"
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.checked_mul(1024)?);
        }
    }
    None
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

    /// v0.3-δ slab pool: post-construction state. New pools/high-water
    /// fields start empty so the first allocation in
    /// `try_cold_load_into_host` takes the tail-carve path.
    #[test]
    fn slab_pools_start_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-slab-init-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        assert!(pager.host_pools.is_empty(), "host_pools should be empty on construction");
        assert_eq!(pager.host_high_water, 0, "host_high_water should be 0 on construction");
        let _ = std::fs::remove_file(&path);
    }

    /// v0.3-δ eviction: popping the LRU returns the offset to the
    /// matching size pool. This is the core "same-size reuse" property
    /// that makes MoE workloads (uniform expert sizes) hit the pool fast
    /// path on every page-in after the first arena fill.
    #[test]
    fn evict_one_recycles_offset_into_size_pool() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-slab-evict-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let mut pager =
            WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();

        // Inject a synthetic resident slot — bypasses ensure_host_arena
        // (no Gpu available in unit tests). The eviction path only touches
        // bookkeeping, not the arena bytes.
        let id = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::GateUp };
        let bytes: u64 = 4096;
        pager.host_resident.insert(id, HostResident { offset: 0, bytes });
        pager.host_lru.push_back(id);
        pager.host_used_bytes = bytes;
        pager.host_high_water = bytes as usize;

        assert!(pager.evict_one_host_lru(), "evict_one should pop the LRU front");
        assert!(!pager.host_resident.contains_key(&id), "residency should be cleared");
        assert_eq!(pager.host_used_bytes, 0, "used bytes should drop by the slot size");
        // High water doesn't recede — that's the v0.3-δ design (only
        // matching-size carves see this; mismatched-size allocs need eviction).
        assert_eq!(pager.host_high_water, bytes as usize, "high water never recedes");
        let pool = pager.host_pools.get(&(bytes as usize)).expect("size pool created");
        assert_eq!(pool, &vec![0usize], "freed offset is in the matching size pool");

        let _ = std::fs::remove_file(&path);
    }

    /// v0.3-δ eviction: LRU order is FIFO (front = least-recently-used).
    /// Two slots inserted in order A, B → eviction pops A first.
    #[test]
    fn evict_one_respects_lru_order() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-slab-lru-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let mut pager =
            WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();

        let a = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::GateUp };
        let b = WeightId::Expert { layer: 0, expert: 1, role: ExpertRole::GateUp };
        let bytes: u64 = 2048;
        pager.host_resident.insert(a, HostResident { offset: 0, bytes });
        pager.host_lru.push_back(a);
        pager.host_resident.insert(b, HostResident { offset: bytes as usize, bytes });
        pager.host_lru.push_back(b);
        pager.host_used_bytes = bytes * 2;
        pager.host_high_water = (bytes as usize) * 2;

        assert!(pager.evict_one_host_lru());
        assert!(!pager.host_resident.contains_key(&a), "A (LRU front) evicted first");
        assert!(pager.host_resident.contains_key(&b), "B survives");

        assert!(pager.evict_one_host_lru());
        assert!(!pager.host_resident.contains_key(&b), "B evicted next");

        assert!(!pager.evict_one_host_lru(), "empty LRU returns false");

        let _ = std::fs::remove_file(&path);
    }

    /// v0.3-δ free_all: clears the new slab state along with the rest.
    /// Doesn't actually call free_all (needs a Gpu), but checks the
    /// fields it would touch start at the expected reset values after
    /// construction — so any future regression that adds non-zero
    /// init would surface here.
    #[test]
    fn slab_state_reset_invariant() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-slab-reset-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        assert_eq!(pager.host_used_bytes(), 0);
        assert!(pager.host_pools.is_empty());
        assert_eq!(pager.host_high_water, 0);
        assert!(pager.host_lru.is_empty());
        assert_eq!(pager.host_resident_count(), 0);
        let _ = std::fs::remove_file(&path);
    }
}
