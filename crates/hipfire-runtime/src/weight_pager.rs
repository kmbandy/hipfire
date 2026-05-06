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

use std::collections::{BTreeMap, HashMap, VecDeque};
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

    /// **Batched parallel host reads** (v0.4). Issue every read in
    /// `batch` and wait for all to complete. Default impl is sequential
    /// (just calls `read_to_host` per item — preserves correctness for
    /// transports that haven't bothered to override). Parallel-capable
    /// transports (`IoUringHostTransport`) override with a single
    /// batched submission, lifting effective drive bandwidth from the
    /// QD=1 ceiling toward the drive's rated parallel throughput.
    ///
    /// Tuple format: `(file_byte_range, host_destination_slice)`.
    /// Caller guarantees:
    /// - `dst.len() == range.len` for every (range, dst) pair
    /// - destination slices do not overlap each other
    fn read_to_host_par(
        &mut self,
        batch: &mut [(ByteRange, &mut [u8])],
    ) -> HipResult<()> {
        for (range, dst) in batch.iter_mut() {
            self.read_to_host(range.offset, range.len, dst)?;
        }
        Ok(())
    }

    /// **v0.5 capability query**: does this transport support reading
    /// directly into VRAM, bypassing the host arena entirely? Returns
    /// `true` for `IoUringP2PTransport` (NVMe → VRAM via dma_buf).
    /// Returns `false` for everything else (pread, io_uring, future
    /// transports that need host staging). The pager dispatches on
    /// this to choose between the host-arena path and the direct path.
    fn supports_direct_to_vram(&self) -> bool {
        false
    }

    /// **v0.5 batched direct-to-VRAM**: read every (range, dst) pair
    /// in `batch` straight into the destination GPU tensors, no host
    /// hop. Default impl errors — only transports that override
    /// `supports_direct_to_vram` to return `true` should be called
    /// here. The pager guards on `supports_direct_to_vram` before
    /// dispatching.
    ///
    /// Tuple format: `(file_byte_range, destination_gpu_tensor)`.
    /// Caller guarantees:
    /// - `dst.buf.size() >= range.len` for every pair
    /// - destination buffers do not overlap each other
    fn read_to_device_par(
        &mut self,
        batch: &mut [(ByteRange, &GpuTensor)],
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        let _ = (batch, gpu);
        Err(hip_bridge::HipError::new(
            0,
            "transport does not support read_to_device_par; pager should not have dispatched here",
        ))
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
// IoUringHostTransport (v0.4)
// ---------------------------------------------------------------------------

/// **v0.4 transport: io_uring batched reads into the host arena.**
///
/// `PreadH2DTransport` tops out at single-threaded queue-depth-1
/// sequential reads — ~500 MB/s on an SN850X-class drive even though
/// the drive can deliver multi-GB/s with parallel I/O. This transport
/// submits N reads in one syscall via io_uring's submission queue,
/// then drains N completions in one syscall — letting the kernel's
/// NVMe driver schedule the I/O in parallel. For our 3-weights-per-
/// layer batches that's effective QD=3 per submission; the ring is
/// sized for headroom (default 32 SQEs) so deeper batching is possible
/// without resizing.
///
/// Linux 5.1+ (we use only basic IORING_OP_READ ops). Reads are
/// **non-O_DIRECT** through the page cache — same semantics as
/// `PreadH2DTransport`, just much more efficient I/O scheduling.
/// A future v0.5+ `IoUringP2PTransport` would add `dma_buf`-mapped
/// VRAM destinations for true NVMe→VRAM with no host hop; that's
/// out of scope here.
#[cfg(target_os = "linux")]
pub struct IoUringHostTransport {
    file: File,
    path: std::path::PathBuf,
    ring: io_uring::IoUring,
    /// Reusable host staging buffer for the single-shot `fetch` /
    /// `fill_into` paths that need a host hop before going to GPU.
    /// Mirrors `PreadH2DTransport::staging`.
    staging: Vec<u8>,
    next_handle: u64,
}

#[cfg(target_os = "linux")]
impl IoUringHostTransport {
    /// Open the HFQ file and create an io_uring with the default
    /// queue depth (32 SQE slots). Plenty for per-layer 3-weight
    /// batches with headroom.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        Self::open_with_depth(path, 32)
    }

    /// Open with a custom queue depth. Larger = more parallel reads
    /// per submission, more ring memory. Power-of-2 recommended per
    /// io_uring conventions.
    pub fn open_with_depth(path: &Path, queue_depth: u32) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // Same file-wide hint as PreadH2D — disables auto-readahead
        // based on sequential heuristics. WILLNEED hints (issued via
        // `advise_prefetch`) still work independently for explicit
        // pre-warming.
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
            }
        }
        let ring = io_uring::IoUring::new(queue_depth)?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
            ring,
            staging: Vec::new(),
            next_handle: 0,
        })
    }

    /// Path the transport was opened with. Useful for diagnostics.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn next_handle(&mut self) -> TransferHandle {
        let h = TransferHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    /// Submit a single read SQE and wait for its completion. Used by
    /// the single-shot `fetch` / `fill_into` / `read_to_host` paths.
    fn iouring_read_one(
        &mut self,
        dst_ptr: *mut u8,
        len: usize,
        file_offset: usize,
    ) -> HipResult<()> {
        use io_uring::{opcode, types};
        use std::os::unix::io::AsRawFd;

        let read_op = opcode::Read::new(
            types::Fd(self.file.as_raw_fd()),
            dst_ptr,
            len as u32,
        )
        .offset(file_offset as u64)
        .build()
        .user_data(0);

        // Safety: caller guarantees dst_ptr/len is a valid mutable
        // region for `len` bytes; fd is valid for self's lifetime.
        unsafe {
            self.ring.submission().push(&read_op).map_err(|_| {
                hip_bridge::HipError::new(0, "io_uring SQ full (single read)")
            })?;
        }
        self.ring.submit_and_wait(1).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("io_uring submit_and_wait (single): {e}"))
        })?;

        let cqe = self.ring.completion().next().ok_or_else(|| {
            hip_bridge::HipError::new(0, "io_uring CQE missing for single read")
        })?;
        let result = cqe.result();
        if result < 0 || result as usize != len {
            return Err(hip_bridge::HipError::new(0, &format!(
                "io_uring read returned {result} (expected {len} bytes at offset {file_offset})"
            )));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Transport for IoUringHostTransport {
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)> {
        if self.staging.len() < len {
            self.staging.resize(len, 0);
        }
        let staging_ptr = self.staging.as_mut_ptr();
        self.iouring_read_one(staging_ptr, len, hfq_offset)?;
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
        if self.staging.len() < len {
            self.staging.resize(len, 0);
        }
        let staging_ptr = self.staging.as_mut_ptr();
        self.iouring_read_one(staging_ptr, len, hfq_offset)?;
        gpu.hip.memcpy_htod(&dst.buf, &self.staging[..len])
    }

    fn read_to_host(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &mut [u8],
    ) -> HipResult<()> {
        debug_assert_eq!(dst.len(), len);
        let dst_ptr = dst.as_mut_ptr();
        self.iouring_read_one(dst_ptr, len, hfq_offset)
    }

    fn wait(&mut self, _handles: &[TransferHandle]) -> HipResult<()> {
        // Reads complete inside the iouring_read_* helpers (they call
        // submit_and_wait). No deferred completion state to drain.
        Ok(())
    }

    /// **The win.** Submit every read in `batch` as one SQE-per-read
    /// in a single push pass, submit them all in one syscall, drain
    /// every CQE in one pass. CQEs may complete out of submission
    /// order; we use `user_data` tags to correlate per-request and
    /// validate each completion against its original byte length.
    fn read_to_host_par(
        &mut self,
        batch: &mut [(ByteRange, &mut [u8])],
    ) -> HipResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        use io_uring::{opcode, types};
        use std::os::unix::io::AsRawFd;
        let fd = self.file.as_raw_fd();

        // Phase 1: push all SQEs onto the submission queue.
        // (Scoped to release the SubmissionQueue borrow before submit.)
        let n = batch.len();
        {
            let mut sq = self.ring.submission();
            for (i, (range, dst)) in batch.iter_mut().enumerate() {
                debug_assert_eq!(dst.len(), range.len);
                let read_op = opcode::Read::new(
                    types::Fd(fd),
                    dst.as_mut_ptr(),
                    range.len as u32,
                )
                .offset(range.offset as u64)
                .build()
                .user_data(i as u64);
                // Safety: caller guarantees dst slices are valid for
                // their declared lengths and don't overlap; fd is
                // valid for self's lifetime.
                unsafe {
                    sq.push(&read_op).map_err(|_| {
                        hip_bridge::HipError::new(0, &format!(
                            "io_uring SQ full at batch idx {i}/{n}"
                        ))
                    })?;
                }
            }
        }

        // Phase 2: one syscall to submit and wait for all completions.
        self.ring.submit_and_wait(n).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("io_uring batch submit_and_wait: {e}"))
        })?;

        // Phase 3: drain exactly N completions, validate each. Snapshot
        // the (offset, len) per slot before this borrow so we can reference
        // them while the CQE iterator holds &mut self.ring transitively.
        let expected_per_idx: Vec<(usize, usize)> = batch.iter()
            .map(|(r, _)| (r.offset, r.len))
            .collect();
        let mut cq = self.ring.completion();
        for _ in 0..n {
            let cqe = cq.next().ok_or_else(|| {
                hip_bridge::HipError::new(0, "io_uring batch CQE missing")
            })?;
            let idx = cqe.user_data() as usize;
            if idx >= n {
                return Err(hip_bridge::HipError::new(0, &format!(
                    "io_uring batch CQE has bogus user_data {idx} (batch size {n})"
                )));
            }
            let result = cqe.result();
            let (file_off, expected) = expected_per_idx[idx];
            if result < 0 || result as usize != expected {
                return Err(hip_bridge::HipError::new(0, &format!(
                    "io_uring batch read [{idx}] returned {result} (expected {expected} bytes at offset {file_off})"
                )));
            }
        }
        Ok(())
    }

    fn advise_prefetch(&self, ranges: &[ByteRange]) {
        // Same as PreadH2D — POSIX_FADV_WILLNEED. io_uring does have
        // IORING_OP_FADVISE but adds plumbing for marginal value here.
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self.file.as_raw_fd();
            for r in ranges {
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
// IoUringP2PTransport (v0.5) — NVMe → VRAM direct via dma_buf
// ---------------------------------------------------------------------------

/// **v0.5 transport: true NVMe → VRAM zero-copy via dma_buf + io_uring.**
///
/// The v0.4 [`IoUringHostTransport`] still routes data through a host
/// arena: NVMe → host pinned memory → memcpy_htod → VRAM. That's a
/// PCIe round-trip per cold-load (NVMe→RAM at ~3 GB/s, then RAM→VRAM
/// at ~25 GB/s). With dma_buf-based P2P the host hop disappears
/// entirely: NVMe DMAs straight into VRAM via the PCIe peer-to-peer
/// path on supported hardware.
///
/// **How it works**:
/// 1. The destination GPU tensor's underlying VRAM buffer is exported
///    as a Linux `dma_buf` fd via `hsa_amd_portable_export_dmabuf`
///    (AMD HSA API, present in ROCm 5.3+, validated end-to-end on
///    ROCm 7.2.2 + R9700 via `examples/dmabuf_probe`).
/// 2. The fd is `mmap`'d into our process address space — yielding a
///    userspace VA that's backed by VRAM physical pages.
/// 3. io_uring submits a `IORING_OP_READ` SQE with that VA as
///    destination. The kernel resolves VA → dma_buf → VRAM physical
///    pages, sets up DMA, and the NVMe driver writes results
///    directly into VRAM (P2P) when the PCIe topology allows it. On
///    topologies that block direct P2P (e.g. devices behind a TB3
///    controller with ACS upstream-forward enabled), the kernel
///    transparently bounces through host RAM at the kernel level —
///    same userspace code, fewer copies than v0.4 either way.
/// 4. After the read completes, we `munmap` the userspace mapping and
///    `hsa_amd_portable_close_dmabuf` the fd. The VRAM allocation
///    itself is untouched and remains owned by HIP.
///
/// **Linux 5.6+** (io_uring + mmap'd dma_buf path). Defers gracefully
/// to v0.4 host-staged reads on Windows/Mac (transport falls back).
///
/// **Per-call overhead**: export+mmap+munmap+close is roughly
/// 4 syscalls per cold-load. Negligible vs the I/O cost (47 MB at
/// drive bandwidth = ~10-15 ms; 4 syscalls add ~10 µs). For workloads
/// that re-use the same VRAM buffers across many reads (e.g. dense
/// scratch ping-pong), a future optimization would cache the
/// dma_buf+mmap and pass the VA directly without re-exporting; for
/// v0.5-α we keep it simple.
#[cfg(target_os = "linux")]
pub struct IoUringP2PTransport {
    file: File,
    path: std::path::PathBuf,
    ring: io_uring::IoUring,
    /// Reusable host staging buffer for the host-tier compat methods
    /// (`fetch`, `fill_into`, `read_to_host`) that callers might still
    /// use. The whole point of P2P is to bypass these, but we keep
    /// them functional so this transport is a drop-in for the others.
    staging: Vec<u8>,
    next_handle: u64,
    /// dlopen handle for libhsa-runtime64.so.1. Kept alive for the
    /// transport's lifetime so the cached function pointers stay
    /// valid. Closed via `libc::dlclose` in `Drop`.
    libhsa: *mut libc::c_void,
    /// `hsa_amd_portable_export_dmabuf` cached at open time.
    hsa_export_fn: HsaExportDmaBufFn,
    /// `hsa_amd_portable_close_dmabuf` cached at open time.
    hsa_close_fn: HsaCloseDmaBufFn,
}

#[cfg(target_os = "linux")]
type HsaExportDmaBufFn = unsafe extern "C" fn(
    *const libc::c_void,
    usize,
    *mut i32,
    *mut u64,
) -> u32;

#[cfg(target_os = "linux")]
type HsaCloseDmaBufFn = unsafe extern "C" fn(i32) -> u32;

// libhsa is dlopen'd once at construction, function pointers are stable
// for the process lifetime. Safe to send the transport across threads.
#[cfg(target_os = "linux")]
unsafe impl Send for IoUringP2PTransport {}

#[cfg(target_os = "linux")]
impl IoUringP2PTransport {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        Self::open_with_depth(path, 32)
    }

    pub fn open_with_depth(path: &Path, queue_depth: u32) -> std::io::Result<Self> {
        let file = File::open(path)?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
            }
        }
        let ring = io_uring::IoUring::new(queue_depth)?;

        // dlopen libhsa-runtime64 and resolve the dma_buf export/close
        // functions. Returns NotFound if the library or the symbols
        // aren't there — caller should fall back to a non-P2P transport
        // on that error.
        let lib_name = std::ffi::CString::new("libhsa-runtime64.so.1").unwrap();
        let libhsa = unsafe { libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
        if libhsa.is_null() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "dlopen libhsa-runtime64.so.1 failed (is ROCm installed?)",
            ));
        }

        let export_sym = std::ffi::CString::new("hsa_amd_portable_export_dmabuf").unwrap();
        let export_ptr = unsafe { libc::dlsym(libhsa, export_sym.as_ptr()) };
        if export_ptr.is_null() {
            unsafe { libc::dlclose(libhsa); }
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "dlsym hsa_amd_portable_export_dmabuf failed (ROCm too old? Need 5.3+)",
            ));
        }
        let hsa_export_fn: HsaExportDmaBufFn = unsafe { std::mem::transmute(export_ptr) };

        let close_sym = std::ffi::CString::new("hsa_amd_portable_close_dmabuf").unwrap();
        let close_ptr = unsafe { libc::dlsym(libhsa, close_sym.as_ptr()) };
        if close_ptr.is_null() {
            unsafe { libc::dlclose(libhsa); }
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "dlsym hsa_amd_portable_close_dmabuf failed",
            ));
        }
        let hsa_close_fn: HsaCloseDmaBufFn = unsafe { std::mem::transmute(close_ptr) };

        Ok(Self {
            file,
            path: path.to_path_buf(),
            ring,
            staging: Vec::new(),
            next_handle: 0,
            libhsa,
            hsa_export_fn,
            hsa_close_fn,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn next_handle(&mut self) -> TransferHandle {
        let h = TransferHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    /// Submit one io_uring read to a raw destination pointer + wait.
    /// Used by both the host-tier compat path (staging buffer) and
    /// the P2P path (mmap'd VRAM via dma_buf).
    fn iouring_read_one(
        &mut self,
        dst_ptr: *mut u8,
        len: usize,
        file_offset: usize,
    ) -> HipResult<()> {
        use io_uring::{opcode, types};
        use std::os::unix::io::AsRawFd;

        let read_op = opcode::Read::new(
            types::Fd(self.file.as_raw_fd()),
            dst_ptr,
            len as u32,
        )
        .offset(file_offset as u64)
        .build()
        .user_data(0);

        unsafe {
            self.ring.submission().push(&read_op).map_err(|_| {
                hip_bridge::HipError::new(0, "p2p io_uring SQ full (single read)")
            })?;
        }
        self.ring.submit_and_wait(1).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("p2p io_uring submit_and_wait: {e}"))
        })?;

        let cqe = self.ring.completion().next().ok_or_else(|| {
            hip_bridge::HipError::new(0, "p2p io_uring CQE missing")
        })?;
        let result = cqe.result();
        if result < 0 || result as usize != len {
            return Err(hip_bridge::HipError::new(0, &format!(
                "p2p io_uring read returned {result} (expected {len} at offset {file_offset})"
            )));
        }
        Ok(())
    }

    /// **The P2P core**: export `vram_ptr` as dma_buf, mmap it, read
    /// `len` bytes from `file_offset` into the mapping, then unmap and
    /// close. The mmap'd VA is the read destination passed to
    /// io_uring; the kernel handles the dma_buf → VRAM resolution.
    fn dmabuf_direct_read(
        &mut self,
        vram_ptr: *mut libc::c_void,
        len: usize,
        file_offset: usize,
    ) -> HipResult<()> {
        // 1. Export VRAM as dma_buf fd.
        let mut dmabuf_fd: i32 = -1;
        let mut dmabuf_offset: u64 = 0;
        let status = unsafe {
            (self.hsa_export_fn)(
                vram_ptr as *const libc::c_void,
                len,
                &mut dmabuf_fd as *mut i32,
                &mut dmabuf_offset as *mut u64,
            )
        };
        if status != 0 || dmabuf_fd < 0 {
            return Err(hip_bridge::HipError::new(0, &format!(
                "hsa_amd_portable_export_dmabuf failed: status=0x{status:04x} fd={dmabuf_fd}"
            )));
        }

        // 2. mmap the dma_buf to get a userspace VA backed by VRAM.
        //    PROT_WRITE because io_uring writes our read data into it;
        //    MAP_SHARED so kernel sees writes go through to VRAM.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dmabuf_fd,
                dmabuf_offset as i64,
            )
        };
        if mapped == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error();
            unsafe { (self.hsa_close_fn)(dmabuf_fd); }
            return Err(hip_bridge::HipError::new(0, &format!(
                "mmap dmabuf fd {dmabuf_fd}: {errno}"
            )));
        }

        // 3. io_uring read into the mapped VA. The kernel resolves the
        //    VA → dma_buf → VRAM physical pages and sets up direct DMA
        //    (or transparently bounces if the topology doesn't allow
        //    P2P). Either way, no userspace copy.
        let read_result = self.iouring_read_one(mapped as *mut u8, len, file_offset);

        // 4. Cleanup, regardless of read outcome.
        let munmap_rc = unsafe { libc::munmap(mapped, len) };
        let close_status = unsafe { (self.hsa_close_fn)(dmabuf_fd) };

        // Surface the read error first; cleanup errors are secondary.
        read_result?;
        if munmap_rc != 0 {
            return Err(hip_bridge::HipError::new(0, &format!(
                "munmap dmabuf VA failed: {}", std::io::Error::last_os_error()
            )));
        }
        if close_status != 0 {
            return Err(hip_bridge::HipError::new(0, &format!(
                "hsa_amd_portable_close_dmabuf returned 0x{close_status:04x}"
            )));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for IoUringP2PTransport {
    fn drop(&mut self) {
        if !self.libhsa.is_null() {
            unsafe { libc::dlclose(self.libhsa); }
        }
    }
}

#[cfg(target_os = "linux")]
impl Transport for IoUringP2PTransport {
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)> {
        // For one-shot fetch: allocate VRAM, then go direct via dma_buf.
        // Same end state as PreadH2D + memcpy_htod, just no host hop.
        let buf = gpu.hip.malloc(len)?;
        let vram_ptr = buf.as_ptr();
        self.dmabuf_direct_read(vram_ptr, len, hfq_offset)?;
        let tensor = GpuTensor {
            buf,
            shape: vec![len],
            dtype: rdna_compute::DType::Raw,
        };
        Ok((tensor, self.next_handle()))
    }

    fn fill_into(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &GpuTensor,
        _gpu: &mut Gpu,
    ) -> HipResult<()> {
        // Direct read into the caller's existing VRAM buffer. This is
        // the dense-paging fast path (gate/up/down scratch ping-pong).
        let vram_ptr = dst.buf.as_ptr();
        self.dmabuf_direct_read(vram_ptr, len, hfq_offset)
    }

    fn read_to_host(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &mut [u8],
    ) -> HipResult<()> {
        // Compat path: regular io_uring read into host buffer. The
        // P2P transport doesn't avoid host buffers — it avoids the
        // *bounce* through one. When the caller explicitly wants
        // bytes in host memory (e.g. the v0.3 host tier warmup),
        // we just do a normal io_uring read.
        debug_assert_eq!(dst.len(), len);
        self.iouring_read_one(dst.as_mut_ptr(), len, hfq_offset)
    }

    fn wait(&mut self, _handles: &[TransferHandle]) -> HipResult<()> {
        Ok(())
    }

    fn requires_dma_buf_alloc(&self) -> bool {
        true
    }

    fn supports_direct_to_vram(&self) -> bool {
        true
    }

    /// **The v0.5 batched win**: export N VRAM destinations as
    /// dma_bufs, mmap each, submit all N io_uring reads in one syscall,
    /// drain N completions in one syscall, then unmap+close. Per-layer
    /// dense overhead: 3 export+mmap, 1 submit_and_wait(3), 3
    /// munmap+close — total ~12 syscalls vs ~30+ on the v0.4
    /// host-staged path.
    fn read_to_device_par(
        &mut self,
        batch: &mut [(ByteRange, &GpuTensor)],
        _gpu: &mut Gpu,
    ) -> HipResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        use io_uring::{opcode, types};
        use std::os::unix::io::AsRawFd;
        let n = batch.len();

        // Phase 1: export + mmap each dst's VRAM region. Track per-slot
        // (dmabuf_fd, mapped_va, len) for cleanup; on any failure we
        // unwind everything allocated so far.
        struct SlotMapping {
            dmabuf_fd: i32,
            mapped: *mut libc::c_void,
            len: usize,
        }
        let mut mappings: Vec<SlotMapping> = Vec::with_capacity(n);

        let cleanup = |mappings: &[SlotMapping], close_fn: HsaCloseDmaBufFn| {
            for m in mappings {
                if !m.mapped.is_null() && m.mapped != libc::MAP_FAILED {
                    unsafe { libc::munmap(m.mapped, m.len); }
                }
                if m.dmabuf_fd >= 0 {
                    unsafe { close_fn(m.dmabuf_fd); }
                }
            }
        };

        for (i, (range, dst)) in batch.iter().enumerate() {
            let vram_ptr = dst.buf.as_ptr();
            let mut dmabuf_fd: i32 = -1;
            let mut dmabuf_offset: u64 = 0;
            let status = unsafe {
                (self.hsa_export_fn)(
                    vram_ptr as *const libc::c_void,
                    range.len,
                    &mut dmabuf_fd as *mut i32,
                    &mut dmabuf_offset as *mut u64,
                )
            };
            if status != 0 || dmabuf_fd < 0 {
                cleanup(&mappings, self.hsa_close_fn);
                return Err(hip_bridge::HipError::new(0, &format!(
                    "p2p batch[{i}/{n}] export failed: status=0x{status:04x} fd={dmabuf_fd}"
                )));
            }
            let mapped = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    range.len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    dmabuf_fd,
                    dmabuf_offset as i64,
                )
            };
            if mapped == libc::MAP_FAILED {
                let errno = std::io::Error::last_os_error();
                unsafe { (self.hsa_close_fn)(dmabuf_fd); }
                cleanup(&mappings, self.hsa_close_fn);
                return Err(hip_bridge::HipError::new(0, &format!(
                    "p2p batch[{i}/{n}] mmap failed: {errno}"
                )));
            }
            mappings.push(SlotMapping { dmabuf_fd, mapped, len: range.len });
        }

        // Phase 2: push all SQEs.
        let fd = self.file.as_raw_fd();
        {
            let mut sq = self.ring.submission();
            for (i, (m, (range, _))) in mappings.iter().zip(batch.iter()).enumerate() {
                let read_op = opcode::Read::new(
                    types::Fd(fd),
                    m.mapped as *mut u8,
                    range.len as u32,
                )
                .offset(range.offset as u64)
                .build()
                .user_data(i as u64);
                unsafe {
                    if sq.push(&read_op).is_err() {
                        drop(sq);
                        cleanup(&mappings, self.hsa_close_fn);
                        return Err(hip_bridge::HipError::new(0, &format!(
                            "p2p io_uring SQ full at batch idx {i}/{n}"
                        )));
                    }
                }
            }
        }

        // Phase 3: submit + wait for all completions.
        if let Err(e) = self.ring.submit_and_wait(n) {
            cleanup(&mappings, self.hsa_close_fn);
            return Err(hip_bridge::HipError::new(0, &format!(
                "p2p batch submit_and_wait: {e}"
            )));
        }

        // Phase 4: drain N CQEs, validate each. Snapshot per-slot
        // (offset, len) before borrowing the completion queue.
        let expected_per_idx: Vec<(usize, usize)> = batch.iter()
            .map(|(r, _)| (r.offset, r.len))
            .collect();
        let mut cq = self.ring.completion();
        let mut drain_err: Option<hip_bridge::HipError> = None;
        for _ in 0..n {
            let cqe = match cq.next() {
                Some(c) => c,
                None => {
                    drain_err = Some(hip_bridge::HipError::new(0, "p2p batch CQE missing"));
                    break;
                }
            };
            let idx = cqe.user_data() as usize;
            if idx >= n {
                drain_err = Some(hip_bridge::HipError::new(0, &format!(
                    "p2p batch CQE bogus user_data {idx}"
                )));
                break;
            }
            let (file_off, expected) = expected_per_idx[idx];
            let result = cqe.result();
            if result < 0 || result as usize != expected {
                drain_err = Some(hip_bridge::HipError::new(0, &format!(
                    "p2p batch read [{idx}] returned {result} (expected {expected} at offset {file_off})"
                )));
                break;
            }
        }
        drop(cq);

        // Phase 5: cleanup all mappings, regardless of drain outcome.
        cleanup(&mappings, self.hsa_close_fn);

        if let Some(e) = drain_err {
            return Err(e);
        }
        Ok(())
    }

    fn advise_prefetch(&self, ranges: &[ByteRange]) {
        // Same as IoUringHostTransport — POSIX_FADV_WILLNEED. Helps
        // when the read happens to land in a path that benefits from
        // page cache priming (typically the host-tier compat methods,
        // not the P2P path).
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self.file.as_raw_fd();
            for r in ranges {
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
    /// LRU recency tracking, implemented as a paired `id ↔ generation`
    /// index so all ops are O(log n).
    ///
    /// **v0.2 policy choice.** Flat LRU across all `WeightId`s. Per PR #167's
    /// Phase 0 measurements, A3B routing entropy is 7.0–7.26 bits out of 8
    /// — so a 32-expert resident cache covers only 42–48 % of routing
    /// decisions, and any fixed-set policy (LRU, MRU, LFU) approaches
    /// random replacement in that regime. Per-layer segmentation is **not**
    /// applied here because the pager has no notion of "current layer" —
    /// it's a passive cache, scheduled by the caller; segmentation requires
    /// a layer-rotation hint that doesn't exist in the v0.2 API. The
    /// headline win in v0.2 comes from io_uring batching the cold-load
    /// path (9.1× prefill on 27B dense), which makes eviction-policy
    /// quality structurally less load-bearing than transport bandwidth.
    /// v0.3 will revisit policy with empirical hit-rate data from
    /// production workloads.
    ///
    /// **Data structure.** A monotonic `lru_gen_counter` mints a fresh
    /// generation on every touch. `lru_id_to_gen` maps id → its current
    /// generation; `lru_gen_to_id` is the reverse, kept sorted by gen so
    /// `pop_first` returns the LRU victim in O(log n). Replaces the v0.1
    /// `VecDeque<WeightId>` whose `touch_lru` was O(n) — for A3B at full
    /// residency (~14k entries) the 448 touches per forward-pass dropped
    /// from ~6M ops to ~6k ops on the critical path.
    lru_id_to_gen: HashMap<WeightId, u64>,
    lru_gen_to_id: BTreeMap<u64, WeightId>,
    lru_gen_counter: u64,
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
            lru_id_to_gen: HashMap::new(),
            lru_gen_to_id: BTreeMap::new(),
            lru_gen_counter: 0,
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

    /// **v0.4 convenience**: open `hfq_path` with the io_uring batched
    /// transport. Linux-only. Use when the workload involves repeated
    /// cold-loads (sub-working-set host budget) where the QD=1 ceiling
    /// of pread becomes the bottleneck.
    #[cfg(target_os = "linux")]
    pub fn with_iouring_transport(hfq_path: &Path, config: PagerConfig) -> std::io::Result<Self> {
        let transport = IoUringHostTransport::open(hfq_path)?;
        Ok(Self::new(Box::new(transport), config))
    }

    /// **v0.5 convenience**: open `hfq_path` with the io_uring + dma_buf
    /// P2P transport. Linux-only, AMD ROCm 5.3+, requires
    /// `CONFIG_PCI_P2PDMA=y` and PCIe ACS settings allowing peer DMA
    /// between the NVMe and the GPU. Falls back to kernel-level bounce
    /// when direct P2P isn't available, but the userspace path is the
    /// same in both cases (no host arena needed).
    #[cfg(target_os = "linux")]
    pub fn with_iouring_p2p_transport(hfq_path: &Path, config: PagerConfig) -> std::io::Result<Self> {
        let transport = IoUringP2PTransport::open(hfq_path)?;
        Ok(Self::new(Box::new(transport), config))
    }

    /// **v0.5 capability query**: does the underlying transport support
    /// direct NVMe → VRAM reads (no host hop)? When `true`, callers
    /// can use [`Self::read_batch_direct_to_vram`] and skip the host
    /// arena entirely.
    pub fn supports_direct_to_vram(&self) -> bool {
        self.transport.supports_direct_to_vram()
    }

    /// **v0.5 batched direct-to-VRAM read**: for each (id, dst) pair,
    /// look up the file byte range via the catalog and issue a
    /// parallel batched read straight into the destination VRAM
    /// buffer. Bypasses the host arena entirely — no `host_resident`
    /// updates, no eviction, no mlock concerns.
    ///
    /// Caller must ensure the destination tensors are at least
    /// `range.len` bytes each. Errors if any id is unregistered.
    pub fn read_batch_direct_to_vram(
        &mut self,
        batch: &[(WeightId, &GpuTensor)],
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let mut io_batch: Vec<(ByteRange, &GpuTensor)> = Vec::with_capacity(batch.len());
        for &(id, tensor) in batch {
            let range = self.catalog.get(&id).copied().ok_or_else(|| {
                hip_bridge::HipError::new(0, &format!(
                    "read_batch_direct_to_vram: weight {id:?} not registered"
                ))
            })?;
            io_batch.push((range, tensor));
        }
        self.transport.read_to_device_par(&mut io_batch, gpu)
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

    /// Look up the on-disk byte range for `id`. Returns `None` when
    /// the weight wasn't `register`'d. Used by callers building batches
    /// for [`Self::try_cold_load_batch_into_host`] without walking the
    /// catalog themselves.
    pub fn byte_range_of(&self, id: WeightId) -> Option<ByteRange> {
        self.catalog.get(&id).copied()
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
        self.lru_touch(id);
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
            self.lru_touch(id);
            let idx = self.transfer_events.len();
            self.transfer_events.push(evt);
            return Ok(Some(TransferEventIdx(idx)));
        }

        // Cold + host-miss — fall back to sync fetch. By return-time
        // the bytes are already in VRAM; no event needed.
        let (tensor, _h) = self.transport.fetch(range.offset, range.len, gpu)?;
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(need);
        self.resident.insert(id, Resident { tensor, bytes: need });
        self.lru_touch(id);
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
        self.ensure_host_arena(gpu)?;
        let offset = match self.try_acquire_host_slot(id, range, gpu)? {
            Some(o) => o,
            None => return Ok(false),
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

    /// **v0.4 batched cold-load**: acquire host slots for every id in
    /// `batch`, then issue all the file reads in parallel via
    /// [`Transport::read_to_host_par`]. Single-shot path is just the
    /// 1-element case of this; the win shows up at batch sizes ≥ 2
    /// when the transport supports parallel reads (io_uring).
    ///
    /// Skips ids that are already host-resident, larger than the entire
    /// arena, or hit cross-size saturation. Updates the residency map
    /// (and LRU + used_bytes) for every id that was actually loaded.
    /// Returns the count loaded (skips don't count).
    pub fn try_cold_load_batch_into_host(
        &mut self,
        batch: &[(WeightId, ByteRange)],
        gpu: &mut Gpu,
    ) -> Result<usize, WeightPagerError> {
        if self.config.host_budget_bytes == 0 || batch.is_empty() {
            return Ok(0);
        }
        self.ensure_host_arena(gpu)?;

        // Phase 1: allocate slots sequentially. Skip ids that:
        // - are already host-resident (no work)
        // - can't be allocated (too big or LRU exhausted)
        let mut allocations: Vec<(WeightId, ByteRange, usize)> = Vec::with_capacity(batch.len());
        for &(id, range) in batch {
            if self.host_resident.contains_key(&id) {
                continue;
            }
            match self.try_acquire_host_slot(id, range, gpu)? {
                Some(off) => allocations.push((id, range, off)),
                None => continue,
            }
        }
        if allocations.is_empty() {
            return Ok(0);
        }

        // Phase 2: parallel batched pread. Construct (range, &mut [u8])
        // tuples backed by non-overlapping arena slices.
        //
        // Safety: the slab allocator guarantees the offsets are in
        // [0, arena_cap) and don't overlap each other. The arena buffer
        // outlives the slices because we hold &mut self.host_arena via
        // arena_base, and we don't touch host_arena between constructing
        // the slices and consuming them in read_to_host_par.
        {
            let arena = self.host_arena.as_mut().expect("ensure_host_arena succeeded");
            let arena_base = arena.as_mut_slice().as_mut_ptr();
            let mut chunks: Vec<(ByteRange, &mut [u8])> = allocations
                .iter()
                .map(|&(_, range, off)| {
                    let dst: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(arena_base.add(off), range.len)
                    };
                    (range, dst)
                })
                .collect();
            self.transport.read_to_host_par(&mut chunks)?;
        }

        // Phase 3: update residency map for everything that loaded.
        let n = allocations.len();
        for &(id, range, off) in &allocations {
            let bytes = range.len as u64;
            self.host_used_bytes = self.host_used_bytes.saturating_add(bytes);
            self.host_resident.insert(id, HostResident { offset: off, bytes });
            self.host_lru.push_back(id);
        }
        if self.config.trace {
            eprintln!(
                "[weight_pager] host-load(batch) {} weights — \
                 used {}MB / hi-water {}MB / {} entries",
                n,
                self.host_used_bytes / (1024 * 1024),
                self.host_high_water / (1024 * 1024),
                self.host_resident.len()
            );
        }
        Ok(n)
    }

    /// Acquire a single host arena slot of `range.len` bytes via the
    /// slab allocator (v0.3-δ). Returns `Some(offset)` on success;
    /// `None` when the weight is bigger than the entire arena OR the
    /// arena hits cross-size saturation with an empty LRU. Does NOT
    /// touch `host_used_bytes`, `host_resident`, or `host_lru` —
    /// caller owns the post-fill bookkeeping.
    ///
    /// Caller must have already called `ensure_host_arena`.
    fn try_acquire_host_slot(
        &mut self,
        id: WeightId,
        range: ByteRange,
        gpu: &mut Gpu,
    ) -> Result<Option<usize>, WeightPagerError> {
        let arena_cap = self.config.host_budget_bytes as usize;
        if range.len > arena_cap {
            if self.config.trace {
                eprintln!(
                    "[weight_pager] host-load skip {id:?}: weight {} bytes > arena cap {} bytes",
                    range.len, arena_cap,
                );
            }
            return Ok(None);
        }
        // One-shot stream sync (lazy, only if eviction is needed). An
        // in-flight memcpy_htod_async could be reading a slot we're
        // about to recycle; sync the transfer stream once before any
        // eviction fires so the recycled offset is safe to overwrite.
        let mut transfer_synced = false;
        loop {
            // (1) Pool hit — same-size slot left over from an eviction.
            if let Some(off) = self
                .host_pools
                .get_mut(&range.len)
                .and_then(|v| v.pop())
            {
                return Ok(Some(off));
            }
            // (2) Tail carve — first-time slot creation.
            if self.host_high_water + range.len <= arena_cap {
                let off = self.host_high_water;
                self.host_high_water += range.len;
                return Ok(Some(off));
            }
            // (3) Evict one and retry.
            if !transfer_synced {
                if let Some(stream) = self.transfer_stream.as_ref() {
                    gpu.hip.stream_synchronize(stream)?;
                }
                transfer_synced = true;
            }
            if !self.evict_one_host_lru() {
                // LRU empty: cross-size saturation. Caller falls back
                // to v0.2 disk path for this weight.
                if self.config.trace {
                    eprintln!(
                        "[weight_pager] host-load skip {id:?}: arena saturated with no \
                         size-{} slots and LRU exhausted",
                        range.len,
                    );
                }
                return Ok(None);
            }
        }
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
                .lru_pop_oldest()
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
        self.lru_clear();
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
        }
        self.resident.insert(id, Resident { tensor, bytes });
        self.lru_touch(id);
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(bytes);
    }

    /// Mark `id` as recently used. No-op if not resident.
    pub fn touch_lru(&mut self, id: WeightId) {
        if self.resident.contains_key(&id) {
            self.lru_touch(id);
        }
    }

    /// Internal: bump `id` to the most-recently-used end of the LRU.
    /// Allocates a fresh generation; if `id` was already in the LRU,
    /// removes its prior generation entry first. O(log n).
    #[inline]
    fn lru_touch(&mut self, id: WeightId) {
        if let Some(old_gen) = self.lru_id_to_gen.remove(&id) {
            self.lru_gen_to_id.remove(&old_gen);
        }
        let gen = self.lru_gen_counter;
        self.lru_gen_counter = self.lru_gen_counter.wrapping_add(1);
        self.lru_id_to_gen.insert(id, gen);
        self.lru_gen_to_id.insert(gen, id);
    }

    /// Internal: pop the least-recently-used entry from the LRU. O(log n).
    #[inline]
    fn lru_pop_oldest(&mut self) -> Option<WeightId> {
        let (_gen, id) = self.lru_gen_to_id.pop_first()?;
        self.lru_id_to_gen.remove(&id);
        Some(id)
    }

    /// Internal: drop all LRU entries (called on `free_all`). Generation
    /// counter intentionally not reset — keeping it monotonic across the
    /// lifetime of the pager avoids any risk of re-issuing an already-seen
    /// generation should a stale id linger anywhere.
    #[inline]
    fn lru_clear(&mut self) {
        self.lru_id_to_gen.clear();
        self.lru_gen_to_id.clear();
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

    /// v0.2 GPU LRU: touch reorders correctly, pop returns oldest.
    /// Locks in the BTreeMap-backed replacement of the v0.1 VecDeque.
    #[test]
    fn lru_touch_reorders_and_pop_returns_oldest() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-lru-touch-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let mut pager =
            WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();

        let a = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::GateUp };
        let b = WeightId::Expert { layer: 0, expert: 1, role: ExpertRole::GateUp };
        let c = WeightId::Expert { layer: 0, expert: 2, role: ExpertRole::GateUp };

        pager.lru_touch(a);
        pager.lru_touch(b);
        pager.lru_touch(c);
        // Insertion order A, B, C → oldest is A.
        assert_eq!(pager.lru_pop_oldest(), Some(a), "A is oldest after A,B,C");
        assert_eq!(pager.lru_pop_oldest(), Some(b), "B is now oldest");

        // Re-touch C then add A back → A is now the newest, C is oldest of {C,A}.
        pager.lru_touch(c);
        pager.lru_touch(a);
        assert_eq!(pager.lru_pop_oldest(), Some(c), "C oldest after re-touch C,A");
        assert_eq!(pager.lru_pop_oldest(), Some(a));
        assert_eq!(pager.lru_pop_oldest(), None, "empty after draining");

        let _ = std::fs::remove_file(&path);
    }

    /// Touching an id already in the LRU must replace its prior generation,
    /// not leave a duplicate (which would let the same id pop twice).
    #[test]
    fn lru_touch_does_not_leave_stale_entries() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-lru-stale-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"x").unwrap();
        let mut pager =
            WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();

        let a = WeightId::Expert { layer: 0, expert: 0, role: ExpertRole::GateUp };
        for _ in 0..5 {
            pager.lru_touch(a);
        }
        assert_eq!(pager.lru_id_to_gen.len(), 1, "id-to-gen has exactly one entry");
        assert_eq!(pager.lru_gen_to_id.len(), 1, "gen-to-id has exactly one entry");
        assert_eq!(pager.lru_pop_oldest(), Some(a));
        assert_eq!(pager.lru_pop_oldest(), None, "no stale duplicate");

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
