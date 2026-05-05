//! MAD-94 v0.2 dense paging smoke: load a dense Qwen3.6 model with
//! `paged_dense = true`, run a real prompt through prefill + greedy decode.
//! The dense FFN weights stream from disk via the pager into shared scratch
//! buffers — every layer reuses the same 3 scratch tensors, refilled in
//! place between layers. Validates correctness (coherent output) and
//! exercises the scratch-buffer paging mechanism.
//!
//! Usage:
//!   cargo run --release --features deltanet --example dense_paged_smoke -- \
//!       ~/Downloads/qwen3.6-27b.mq4
//!
//! Environment (mirrors a3b_paged_smoke / a3b_smoke_forward):
//!   HIPFIRE_SMOKE_MODE       — "raw" (default) or "chat" (Qwen3.5 template)
//!   HIPFIRE_SMOKE_PROMPT     — prompt text (default: "Hello")
//!   HIPFIRE_SMOKE_STEPS      — N greedy decode steps (default: 1)
//!   HIPFIRE_SMOKE_KV         — q8 (default) | asym2 | asym3 | asym4
//!   HIPFIRE_SMOKE_KV_SEQ     — KV cache size (default 256)
//!   HIPFIRE_HOST_BUDGET_MB   — pinned-host tier budget (default: 0 = off, v0.2)
//!                              Set above the paged-weight footprint to engage
//!                              the v0.3 host-tier — paging then runs at full
//!                              PCIe bandwidth instead of NVMe.
//!   HIPFIRE_PAGER_TRACE      — when set, pager prints residency events

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dense_paged_smoke <model.mq4>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let n_steps: usize = std::env::var("HIPFIRE_SMOKE_STEPS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);

    eprintln!("Opening: {model_path}");
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let mut config = qwen35::config_from_hfq(&hfq).expect("read config");
    assert!(config.num_experts == 0,
        "this smoke test expects a DENSE model (num_experts=0); got {}", config.num_experts);

    config.paged_dense = true;
    if let Ok(s) = std::env::var("HIPFIRE_HOST_BUDGET_MB") {
        let mb: u64 = s.parse().expect("HIPFIRE_HOST_BUDGET_MB must be a u64");
        config.host_budget_bytes = mb * 1024 * 1024;
        eprintln!("Host (pinned-RAM) tier budget: {mb} MB");
    } else {
        eprintln!("Host (pinned-RAM) tier: disabled (set HIPFIRE_HOST_BUDGET_MB to engage v0.3)");
    }
    eprintln!("paged_dense=true  ({}-layer dense, hidden_dim={}, dim={})",
        config.n_layers, config.hidden_dim, config.dim);

    eprintln!("\nLoading weights (paged dense FFN) ...");
    let mut gpu = rdna_compute::Gpu::init().expect("init gpu");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded.");
    if let Some(scratch) = &weights.dense_scratch {
        let total_scratch = scratch.gate_bytes + scratch.up_bytes + scratch.down_bytes;
        eprintln!("DenseScratch: {} MB total  (gate {} MB, up {} MB, down {} MB; dtype {:?})",
            total_scratch / (1024*1024),
            scratch.gate_bytes / (1024*1024),
            scratch.up_bytes / (1024*1024),
            scratch.down_bytes / (1024*1024),
            scratch.gpu_dtype);
        eprintln!("DenseScratch shapes: gate=[{},{}], down=[{},{}]",
            scratch.gate_m, scratch.gate_k, scratch.down_m, scratch.down_k);
    }
    if let Some(p) = &weights.pager {
        let p = p.borrow();
        eprintln!("Pager: registered {}, vram-resident {}, vram {} bytes; host-resident {}, host {} bytes",
            p.registered_count(), p.resident_count(), p.vram_used_bytes(),
            p.host_resident_count(), p.host_used_bytes());
    }

    let kv_seq = std::env::var("HIPFIRE_SMOKE_KV_SEQ")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(256usize);
    let kv_mode = std::env::var("HIPFIRE_SMOKE_KV").unwrap_or_else(|_| "q8".to_string());
    eprintln!("KV cache mode: {kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
        "asym2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
        _ => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state alloc");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch alloc");

    let prompt_mode = std::env::var("HIPFIRE_SMOKE_MODE").unwrap_or_else(|_| "raw".to_string());
    let user_prompt = std::env::var("HIPFIRE_SMOKE_PROMPT").unwrap_or_else(|_| "Hello".to_string());
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let prompt_tokens: Vec<u32> = if prompt_mode == "chat" {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user = tokenizer.encode("user");
        let asst = tokenizer.encode("assistant");
        let nl = tokenizer.encode("\n");
        let body = tokenizer.encode(&user_prompt);
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&user);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&body);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&asst);
        chat.extend_from_slice(&nl);
        chat
    } else {
        tokenizer.encode(&user_prompt)
    };
    eprintln!("Prompt ({prompt_mode} mode): {} tokens", prompt_tokens.len());

    eprintln!("\n=== forward_scratch prefill (paged dense) ===");
    let t0 = std::time::Instant::now();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, tok, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("forward_scratch failed");
    }
    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
    let elapsed = t0.elapsed();
    let n_prompt = prompt_tokens.len();
    let pf_us = elapsed.as_micros() as f64;
    eprintln!("prefill {} toks in {:.2} ms ({:.1} tok/s)",
        n_prompt, pf_us / 1000.0, (n_prompt as f64) * 1_000_000.0 / pf_us);

    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &logits {
        if v.is_nan() { n_nan += 1; }
        else if v.is_infinite() { n_inf += 1; }
        else {
            if v < min_v { min_v = v; }
            if v > max_v { max_v = v; }
        }
    }
    eprintln!("  logits.len = {}", logits.len());
    eprintln!("  finite range: [{:.4}, {:.4}]", min_v, max_v);
    eprintln!("  NaNs: {n_nan}  Infs: {n_inf}");
    assert!(n_nan == 0 && n_inf == 0,
        "paged dense FFN forward produced NaN/Inf — pager fill or alias is wrong");

    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    indexed.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed[..5].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  top-5 token ids: {:?}", &indexed[..5]);
    let argmax = indexed[0].0;
    let decoded_argmax = tokenizer.decode(&[argmax]);
    eprintln!("  argmax = {argmax} '{}'  (elapsed: {:?})",
        decoded_argmax.replace('\n', "\\n"), elapsed);

    if n_steps > 1 {
        eprintln!("\n=== decoding {} more tokens greedily ===", n_steps - 1);
        let mut next = argmax;
        let base_pos = prompt_tokens.len();
        let mut timings = Vec::with_capacity(n_steps.saturating_sub(1));
        for step in 1..n_steps {
            let t0 = std::time::Instant::now();
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, next, base_pos + step - 1,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward_scratch failed");
            let l = gpu.download_f32(&scratch.logits).expect("download");
            timings.push(t0.elapsed());
            assert!(!l.iter().any(|v| v.is_nan() || v.is_infinite()),
                "NaN/Inf at decode step {step}");
            next = llama::argmax(&l);
            let decoded = tokenizer.decode(&[next]);
            eprintln!("  step {step:2} -> {next:6} '{}'  ({} µs)",
                decoded.replace('\n', "\\n"), timings.last().unwrap().as_micros());
        }

        let settled: Vec<_> = timings.iter().skip(2).collect();
        if !settled.is_empty() {
            let sum: u128 = settled.iter().map(|d| d.as_micros()).sum();
            let avg_us = sum / settled.len() as u128;
            let tok_per_s = 1_000_000.0 / avg_us as f64;
            eprintln!("\nsteady-state decode (n={}): avg {} µs/tok = {:.1} tok/s",
                settled.len(), avg_us, tok_per_s);
        }
    }

    eprintln!("\n=== DENSE PAGED SMOKE TEST PASSED ===");
}
