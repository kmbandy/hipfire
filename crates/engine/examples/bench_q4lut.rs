//! Benchmark Q4_LUT vs Q4_K vs Q8_0 — testing LDS codebook approach.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");

    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = engine::gguf::GgufFile::open(std::path::Path::new(path)).unwrap();
    let ti = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let raw_q4k = gguf.tensor_data(ti);
    let m = 2048usize;
    let k = 2048usize;

    // Convert Q4_K → Q4_LUT (lossless: precompute dequant codebook per sub-block)
    let q4lut = convert_q4k_to_q4lut(raw_q4k, m * k);
    eprintln!("Q4_K:  {} bytes ({:.4} B/w)", raw_q4k.len(), raw_q4k.len() as f64 / (m * k) as f64);
    eprintln!("Q4LUT: {} bytes ({:.4} B/w)", q4lut.len(), q4lut.len() as f64 / (m * k) as f64);

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();

    // Also create Q8_0 synthetic data for comparison
    let a_f32 = engine::llama::dequantize_q4_k(raw_q4k, m * k);
    let q8_data = quantize_q8(&a_f32);

    let d_q4k = gpu.upload_raw(raw_q4k, &[raw_q4k.len()]).unwrap();
    let d_q4lut = gpu.upload_raw(&q4lut, &[q4lut.len()]).unwrap();
    let d_q8 = gpu.upload_raw(&q8_data, &[q8_data.len()]).unwrap();

    let d_y1 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_y2 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_y3 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    let n_warmup = 50;
    let n_iter = 500;

    // Warmup
    for _ in 0..n_warmup {
        gpu.gemv_q4k(&d_q4k, &d_x, &d_y1, m, k).unwrap();
        gpu.gemv_q4lut(&d_q4lut, &d_x, &d_y2, m, k).unwrap();
        gpu.gemv_q8_0(&d_q8, &d_x, &d_y3, m, k).unwrap();
    }

    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();

    // Q4_K
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q4k(&d_q4k, &d_x, &d_y1, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let q4k_bytes = (m * (k / 256) * 144 + k * 4) as f64;
    let bw = q4k_bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    eprintln!("Q4_K:  {:6.1}us  {:6.1} GB/s  {:4.1}% peak",
        ms * 1000.0 / n_iter as f32, bw, bw / 448.0 * 100.0);

    // Q4_LUT
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q4lut(&d_q4lut, &d_x, &d_y2, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let lut_bytes = (m * (k / 32) * 48 + k * 4) as f64;
    let bw = lut_bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    let us = ms * 1000.0 / n_iter as f32;
    eprintln!("Q4LUT: {:6.1}us  {:6.1} GB/s  {:4.1}% peak",
        us, bw, bw / 448.0 * 100.0);

    // Q8_0
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q8_0(&d_q8, &d_x, &d_y3, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let q8_bytes = (m * (k / 32) * 34 + k * 4) as f64;
    let bw = q8_bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    eprintln!("Q8_0:  {:6.1}us  {:6.1} GB/s  {:4.1}% peak",
        ms * 1000.0 / n_iter as f32, bw, bw / 448.0 * 100.0);

    // Verify correctness: Q4_LUT should match Q4_K exactly (lossless conversion)
    let y1 = gpu.download_f32(&d_y1).unwrap();
    let y2 = gpu.download_f32(&d_y2).unwrap();
    let max_diff: f32 = y1.iter().zip(y2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("\nQ4_LUT vs Q4_K max diff: {max_diff:.8}");

    // Check VGPRs
    eprintln!("\nCheck VGPRs: /opt/rocm-6.3.4/lib/llvm/bin/clang-offload-bundler + llvm-objdump on /tmp/hipfire_kernels/gemv_q4lut.hsaco");
}

/// Convert Q4_K blocks to Q4_LUT blocks (lossless: precompute codebook).
/// Q4_K: 144 bytes per 256 elements → Q4_LUT: 8 × 48 bytes per 256 elements = 384 bytes.
fn convert_q4k_to_q4lut(q4k_data: &[u8], n_elements: usize) -> Vec<u8> {
    let q4k_block_bytes = 144;
    let q4k_block_elems = 256;
    let lut_block_bytes = 48; // per 32 elements
    let nblocks = (n_elements + q4k_block_elems - 1) / q4k_block_elems;
    let mut output = vec![0u8; nblocks * 8 * lut_block_bytes]; // 8 sub-blocks per super-block

    for b in 0..nblocks {
        let off = b * q4k_block_bytes;
        if off + q4k_block_bytes > q4k_data.len() { break; }

        let d = engine::llama::f16_to_f32(u16::from_le_bytes([q4k_data[off], q4k_data[off + 1]]));
        let dmin = engine::llama::f16_to_f32(u16::from_le_bytes([q4k_data[off + 2], q4k_data[off + 3]]));

        let sc_data = &q4k_data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        let qdata = &q4k_data[off + 16..off + 16 + 128];

        // 4 groups × 2 sub-blocks = 8 LUT blocks
        for group in 0..4 {
            for sub in 0..2 {
                let sb_idx = group * 2 + sub;
                let eff_scale = d * scales[sb_idx] as f32;
                let eff_min = dmin * mins[sb_idx] as f32;

                let out_off = (b * 8 + sb_idx) * lut_block_bytes;

                // Write codebook: 16 f16 values
                for n in 0..16u8 {
                    let val = eff_scale * n as f32 - eff_min;
                    let f16_val = engine::llama::f32_to_f16(val);
                    let co = out_off + (n as usize) * 2;
                    output[co..co + 2].copy_from_slice(&f16_val.to_le_bytes());
                }

                // Write packed nibbles (16 bytes for 32 elements)
                // Q4_K layout: group*32 bytes, sub=0 uses low nibbles, sub=1 uses high nibbles
                // Q4_LUT layout: byte[i] = low_nibble(elem i) | high_nibble(elem i+16)
                for i in 0..16 {
                    let src_byte = qdata[group * 32 + i + sub * 16];
                    // For sub=0: original byte has elem i(lo) and elem i+32(hi) from Q4_K
                    // But within a 32-element LUT block, we need:
                    //   byte[i] = nibble(elem i) | nibble(elem i+16) << 4
                    // In Q4_K: byte[group*32+l] has lo=elem group*64+l, hi=elem group*64+32+l
                    // For sub=0 (elements 0..31 of group): we want the low nibbles
                    //   elem i → lo nibble of qdata[group*32 + i]
                    //   elem i+16 → lo nibble of qdata[group*32 + i + 16]
                    let nib_lo = if sub == 0 {
                        qdata[group * 32 + i] & 0xF
                    } else {
                        qdata[group * 32 + i] >> 4
                    };
                    let nib_hi = if sub == 0 {
                        qdata[group * 32 + 16 + i] & 0xF
                    } else {
                        qdata[group * 32 + 16 + i] >> 4
                    };
                    output[out_off + 32 + i] = nib_lo | (nib_hi << 4);
                }
            }
        }
    }

    output
}

/// Simple Q8 quantization for comparison
fn quantize_q8(f32_data: &[f32]) -> Vec<u8> {
    let mut output = Vec::new();
    for block in f32_data.chunks(32) {
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let scale_f16 = engine::llama::f32_to_f16(scale);
        output.extend_from_slice(&scale_f16.to_le_bytes());
        for &v in block {
            let q = (v * inv_scale).round().max(-128.0).min(127.0) as i8;
            output.push(q as u8);
        }
    }
    output
}
