//! v0.5 prerequisite probe: verify that AMD's
//! `hsa_amd_portable_export_dmabuf` actually returns a valid Linux
//! dma_buf fd for a HIP-allocated VRAM buffer on this hardware +
//! ROCm install.
//!
//! Does NOT need the engine deltanet feature — pure HIP + dlopen +
//! libhsa-runtime64 probe. Run with:
//!
//!   cargo run --release --example dmabuf_probe
//!
//! Output on success:
//!   - "status: 0 (HSA_STATUS_SUCCESS)"
//!   - dmabuf_fd is a positive integer
//!   - /proc/self/fd/N readlink shows "anon_inode:dmabuf" (or similar)
//!   - fstat reports a regular fd
//!   - close returns 0
//!
//! On failure: the HSA error code tells us what's blocked.
//!   - 0x1001 NOT_INITIALIZED — HSA runtime never came up
//!   - 0x1002 INVALID_ARGUMENT — null pointers or bad params
//!   - 0x1008 INVALID_AGENT   — gfx1201 driver doesn't support export
//!   - 0x100A INVALID_ALLOCATION — pointer not in a single allocation
//!   - 0x1004 OUT_OF_RESOURCES — fd creation failed (rlimit?)

use std::ffi::{CString, c_void};
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::FromRawFd;

fn main() {
    println!("=== v0.5 dma_buf probe ===\n");

    // 1. Init HIP and allocate 16 MB VRAM. HIP initialization brings
    //    HSA up under the hood (libamdhip64 calls hsa_init), so by the
    //    time we have a HIP allocation, libhsa-runtime64 is alive and
    //    ready for the export call.
    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("FAIL: Gpu::init failed: {e}");
            std::process::exit(1);
        }
    };
    println!("[1] Gpu initialized.");

    let alloc_size: usize = 16 * 1024 * 1024;
    let buf = match gpu.hip.malloc(alloc_size) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL: hip malloc {alloc_size} bytes: {e}");
            std::process::exit(1);
        }
    };
    let vram_ptr = buf.as_ptr();
    println!("[2] HIP malloc'd {alloc_size} bytes at {vram_ptr:p}");

    // 2. dlopen libhsa-runtime64 and resolve the export/close symbols.
    //    We go via dlopen rather than #[link] because libhsa-runtime64
    //    is in /opt/rocm/lib which isn't on the default linker path,
    //    and we don't want to add ROCm-specific build.rs config to the
    //    engine crate just for one probe binary.
    let lib_name = CString::new("libhsa-runtime64.so.1").unwrap();
    let lib = unsafe { libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
    if lib.is_null() {
        let err = unsafe { libc::dlerror() };
        let err_msg = if err.is_null() {
            "(no dlerror)".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned() }
        };
        eprintln!("FAIL: dlopen libhsa-runtime64.so.1: {err_msg}");
        std::process::exit(1);
    }
    println!("[3] dlopen libhsa-runtime64.so.1 succeeded.");

    let export_sym = CString::new("hsa_amd_portable_export_dmabuf").unwrap();
    let export_ptr = unsafe { libc::dlsym(lib, export_sym.as_ptr()) };
    if export_ptr.is_null() {
        eprintln!("FAIL: dlsym hsa_amd_portable_export_dmabuf returned null");
        std::process::exit(1);
    }
    type ExportFn =
        unsafe extern "C" fn(*const c_void, usize, *mut i32, *mut u64) -> u32;
    let export_fn: ExportFn = unsafe { std::mem::transmute(export_ptr) };
    println!("[4] resolved hsa_amd_portable_export_dmabuf.");

    let close_sym = CString::new("hsa_amd_portable_close_dmabuf").unwrap();
    let close_ptr = unsafe { libc::dlsym(lib, close_sym.as_ptr()) };
    if close_ptr.is_null() {
        eprintln!("FAIL: dlsym hsa_amd_portable_close_dmabuf returned null");
        std::process::exit(1);
    }
    type CloseFn = unsafe extern "C" fn(i32) -> u32;
    let close_fn: CloseFn = unsafe { std::mem::transmute(close_ptr) };
    println!("[5] resolved hsa_amd_portable_close_dmabuf.");

    // 3. Call export. This is the moment of truth.
    let mut dmabuf_fd: i32 = -1;
    let mut offset: u64 = 0;
    println!("\n[6] calling hsa_amd_portable_export_dmabuf(ptr={vram_ptr:p}, size={alloc_size})...");
    let status = unsafe {
        export_fn(
            vram_ptr as *const c_void,
            alloc_size,
            &mut dmabuf_fd as *mut i32,
            &mut offset as *mut u64,
        )
    };
    println!("    status: 0x{status:04x}");
    println!("    dmabuf_fd: {dmabuf_fd}");
    println!("    offset: {offset}");

    if status != 0 {
        eprintln!("\nFAIL: export returned non-success status 0x{status:04x}");
        eprintln!("  (0x1001=NOT_INITIALIZED, 0x1002=INVALID_ARGUMENT,");
        eprintln!("   0x1008=INVALID_AGENT, 0x100A=INVALID_ALLOCATION,");
        eprintln!("   0x1004=OUT_OF_RESOURCES)");
        std::process::exit(1);
    }
    if dmabuf_fd < 0 {
        eprintln!("\nFAIL: status was success but dmabuf_fd is {dmabuf_fd}");
        std::process::exit(1);
    }

    // 4. Verify the fd is a real dma_buf. On Linux, dma_buf fds are
    //    anonymous inodes — /proc/self/fd/N readlink returns
    //    something like "/anon_inode:dmabuf" or similar.
    println!("\n[7] verifying fd...");
    let link_path = format!("/proc/self/fd/{dmabuf_fd}");
    match std::fs::read_link(&link_path) {
        Ok(target) => println!("    /proc/self/fd/{dmabuf_fd} → {target:?}"),
        Err(e) => println!("    /proc/self/fd/{dmabuf_fd} readlink failed: {e}"),
    }

    // fstat — anonymous inodes still have device/inode/mode reachable
    // via fstat. We use mem::forget on the File at the end so its Drop
    // doesn't double-close (we close via HSA below).
    let file = unsafe { std::fs::File::from_raw_fd(dmabuf_fd) };
    match file.metadata() {
        Ok(meta) => {
            println!(
                "    fstat: dev={} ino={} mode=0o{:o} nlink={} size={}",
                meta.dev(),
                meta.ino(),
                meta.mode(),
                meta.nlink(),
                meta.size()
            );
        }
        Err(e) => println!("    fstat failed: {e}"),
    }
    std::mem::forget(file);

    // 5. Close via HSA's matching API. Don't use libc::close — HSA
    //    tracks the fd internally and may need to release its own
    //    references when we close.
    println!("\n[8] closing via hsa_amd_portable_close_dmabuf({dmabuf_fd})...");
    let close_status = unsafe { close_fn(dmabuf_fd) };
    println!("    close status: 0x{close_status:04x}");
    if close_status != 0 {
        eprintln!("WARN: close returned non-success 0x{close_status:04x}");
    }

    // 6. Free the HIP allocation (drops via the hip_bridge wrapper).
    drop(buf);

    println!("\n=== DMABUF EXPORT PROBE PASSED ===");
    println!("v0.5 implementation can proceed with confidence.");
}
