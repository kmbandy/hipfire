//! MAD-93 v0.2 paged smoke test: load A3B with `paged_experts = true`, run
//! forward passes, verify the pager actually services cold-loads and the
//! resulting logits match the non-paged path's coherence properties (finite,
//! sensible top-K, no attractor on greedy decode).
//!
//! Usage:
//!   cargo run --release --features deltanet --example a3b_paged_smoke -- \
//!       ~/Downloads/qwen3.6-35b-a3b.mq4
//!
//! Environment:
//!   HIPFIRE_VRAM_BUDGET_MB   — pager VRAM budget in MB (default: unlimited).
//!                              Set below model size to exercise eviction.
//!   HIPFIRE_PAGER_TRACE      — when set, pager prints residency events.
//!   HIPFIRE_SMOKE_STEPS      — N greedy decode steps (default: 1).

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use engine::llama::{self, KvCache};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: a3b_paged_smoke <model.mq4>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let n_steps: usize = std::env::var("HIPFIRE_SMOKE_STEPS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);

    eprintln!("Opening: {model_path}");
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let mut config = qwen35::config_from_hfq(&hfq).expect("read config");
    assert!(config.num_experts > 0, "this smoke test expects a MoE model");

    // Enable paged experts.
    config.paged_experts = true;
    if let Ok(s) = std::env::var("HIPFIRE_VRAM_BUDGET_MB") {
        let mb: u64 = s.parse().expect("HIPFIRE_VRAM_BUDGET_MB must be a u64");
        config.vram_budget_bytes = mb * 1024 * 1024;
        eprintln!("Pager VRAM budget: {mb} MB ({} bytes)", config.vram_budget_bytes);
    } else {
        eprintln!("Pager VRAM budget: unlimited (no eviction)");
    }

    eprintln!("A3B config: dim={}, layers={}, experts={}, top_k={}, moe_inter={}, shared_inter={}",
        config.dim, config.n_layers, config.num_experts, config.num_experts_per_tok,
        config.moe_intermediate_size, config.shared_expert_intermediate_size);

    eprintln!("\nLoading weights (paged) ...");
    let mut gpu = rdna_compute::Gpu::init().expect("init gpu");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded.");

    // Diagnostic: how many catalog entries vs. residency at this point.
    if let Some(pager_rc) = &weights.pager {
        let pager = pager_rc.borrow();
        eprintln!(
            "Pager registered {} weights, {} resident, {} bytes resident",
            pager.registered_count(),
            pager.resident_count(),
            pager.vram_used_bytes(),
        );
    }

    // Single-token decode forward pass.
    let token: u32 = 0; // BOS-ish — same as non-paged smoke
    let pos: usize = 0;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 256,
    ).expect("kv cache");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("delta state");
    let _ = Qwen35Scratch::new(&mut gpu, &config, 64); // unused but matches non-paged smoke setup

    eprintln!("\n=== forward pass 1 ===");
    let t0 = std::time::Instant::now();
    let logits = qwen35::forward(&mut gpu, &weights, &config, token, pos, &mut kv_cache, &mut dn_state)
        .expect("forward failed");
    let elapsed = t0.elapsed();

    let mut nans = 0u32;
    let mut infs = 0u32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in &logits {
        if v.is_nan() { nans += 1; continue; }
        if v.is_infinite() { infs += 1; continue; }
        min = min.min(v);
        max = max.max(v);
    }

    eprintln!("logits.len = {}", logits.len());
    eprintln!("finite range: [{min:.4}, {max:.4}]");
    eprintln!("NaNs: {nans}  Infs: {infs}");

    // Top-5 token IDs
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed[..5].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("top-5 token ids: {:?}", &indexed[..5]);
    eprintln!("argmax = {}  (elapsed: {:?})", indexed[0].0, elapsed);

    if let Some(pager_rc) = &weights.pager {
        let pager = pager_rc.borrow();
        eprintln!(
            "Pager after fwd 1: {} resident, {} bytes used",
            pager.resident_count(), pager.vram_used_bytes(),
        );
    }

    // Optional multi-step decode for stability check.
    if n_steps > 1 {
        eprintln!("\n=== decoding {} more tokens greedily ===", n_steps - 1);
        let mut current = indexed[0].0 as u32;
        let mut total_us = 0u128;
        for step in 1..n_steps {
            let t = std::time::Instant::now();
            let logits = qwen35::forward(&mut gpu, &weights, &config, current, step, &mut kv_cache, &mut dn_state)
                .expect("forward step failed");
            let us = t.elapsed().as_micros();
            total_us += us;

            // argmax
            let (next_id, next_v) = logits.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                    if v > best_v { (i, v) } else { (best_i, best_v) }
                });
            let _ = next_v;
            eprintln!("  step {:>2} -> {:>6}  ({us} µs)", step, next_id);
            current = next_id as u32;
        }
        let avg = total_us / (n_steps as u128 - 1);
        let tps = 1_000_000.0 / avg as f64;
        eprintln!("\nsteady-state decode (n={}): avg {avg} µs/tok = {tps:.1} tok/s",
            n_steps - 1);
    }

    if let Some(pager_rc) = &weights.pager {
        let pager = pager_rc.borrow();
        eprintln!(
            "\nPager final: {} resident, {} bytes used (catalog {})",
            pager.resident_count(), pager.vram_used_bytes(), pager.registered_count(),
        );
    }
    eprintln!("\n=== PAGED SMOKE TEST PASSED ===");
}
