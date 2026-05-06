//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

mod ffi;
mod error;
mod kernarg;
mod rocblas;

pub use error::{
    HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    HIP_ERROR_PEER_ACCESS_NOT_ENABLED, HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
pub use ffi::{Event, Function, Graph, GraphExec, HipPointerAttribute, HipRuntime, Module, Stream};
pub use ffi::launch_counters;
pub use kernarg::KernargBlob;
pub use rocblas::{Rocblas, RocblasDatatype, RocblasError, RocblasOperation, RocblasResult};

/// Re-export memory copy direction for callers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// Mirrors `hipMemoryType`. FFI stores raw `u32`; use `from_raw` to convert.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Unregistered = 0,
    Host = 1,
    Device = 2,
    Managed = 3,
    Array = 10,
    Unified = 11,
}

impl MemoryType {
    pub fn from_raw(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Unregistered),
            1 => Some(Self::Host),
            2 => Some(Self::Device),
            3 => Some(Self::Managed),
            10 => Some(Self::Array),
            11 => Some(Self::Unified),
            _ => None,
        }
    }
}

/// Opaque GPU buffer handle. Tracks pointer + size for safety.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Create a non-owning DeviceBuffer from a raw pointer and size.
    /// The caller must ensure the pointer is valid GPU memory.
    /// The resulting buffer must NOT be freed (it doesn't own the memory).
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer { ptr, size }
    }

    /// Create a non-owning alias to the same GPU memory.
    /// The alias must not outlive the original buffer.
    /// Used for reshaping tensors without reallocating.
    /// # Safety
    /// Caller must ensure the alias doesn't outlive the original.
    pub unsafe fn alias(&self) -> DeviceBuffer {
        DeviceBuffer { ptr: self.ptr, size: self.size }
    }
}

// DeviceBuffer is Send — GPU pointers can be sent between threads.
// They are NOT Sync — concurrent access requires stream synchronization.
unsafe impl Send for DeviceBuffer {}

/// Pinned (page-locked) host memory buffer. Allocated via `hipHostMalloc`
/// — the OS guarantees the pages stay resident, which lets the DMA engine
/// move data directly without bouncing through a temporary kernel buffer.
/// Net effect: H2D throughput goes from ~5 GB/s (pageable) to ~25 GB/s
/// (pinned) on PCIe 4.0 — the substrate for the WeightPager v0.3 host tier.
///
/// Caller owns the lifecycle: `HipRuntime::host_malloc` to allocate,
/// `HipRuntime::host_free` to release. Like `DeviceBuffer`, this does
/// **not** auto-drop — the FFI layer requires explicit free so the runtime
/// can return errors and the higher layers can defer destruction past
/// outstanding async transfers.
pub struct HostBuffer {
    ptr: *mut u8,
    size: usize,
}

impl HostBuffer {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Borrow the pinned region as a byte slice. Safe as long as no
    /// concurrent DMA is in flight against the same region (caller's
    /// responsibility — pinned pages are still racy across CPU/GPU).
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Construct from a raw pointer + size without taking ownership.
    /// The resulting buffer must NOT be freed.
    /// # Safety
    /// Caller must ensure `ptr` references valid pinned host memory of
    /// at least `size` bytes that outlives this alias.
    pub unsafe fn from_raw(ptr: *mut u8, size: usize) -> HostBuffer {
        HostBuffer { ptr, size }
    }
}

// HostBuffer is Send — pinned host pointers can be sent between threads.
// Not Sync — concurrent CPU writes during in-flight DMA are unsafe.
unsafe impl Send for HostBuffer {}
