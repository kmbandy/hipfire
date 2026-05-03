//! MAD-93 v0.2 paged smoke test: load A3B with `paged_experts = true`, run
//! a real prompt through prefill + greedy decode, verify the pager actually
//! services cold-loads and the resulting output is coherent (apples-to-apples
//! comparison with `a3b_smoke_forward` — same prompt, same prefill, same
//! decode loop, only `paged_experts` differs).
//!
//! Usage:
//!   cargo run --release --features deltanet --example a3b_paged_smoke -- \
//!       ~/Downloads/qwen3.6-35b-a3b.mq4
//!
//! Environment (matches a3b_smoke_forward where applicable):
//!   HIPFIRE_SMOKE_MODE       — "raw" (default) or "chat" (Qwen3.5 template)
//!   HIPFIRE_SMOKE_PROMPT     — prompt text (default: "Hello")
//!   HIPFIRE_SMOKE_STEPS      — N greedy decode steps (default: 1)
//!   HIPFIRE_SMOKE_KV         — q8 (default) | asym2 | asym3 | asym4
//!   HIPFIRE_SMOKE_KV_SEQ     — KV cache size (default 256)
//!   HIPFIRE_VRAM_BUDGET_MB   — pager VRAM budget in MB (default: unlimited)
//!   HIPFIRE_PAGER_TRACE      — when set, pager prints residency events

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

    // Enable paged experts — the only meaningful difference vs a3b_smoke_forward.
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
    if let Some(p) = &weights.pager {
        let p = p.borrow();
        eprintln!("Pager: registered {}, resident {}, used {} bytes",
            p.registered_count(), p.resident_count(), p.vram_used_bytes());
    }

    // KV cache + DeltaNet state — same setup as a3b_smoke_forward.
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

    // Tokenizer + prompt — exactly mirrors a3b_smoke_forward.
    let prompt_mode = std::env::var("HIPFIRE_SMOKE_MODE").unwrap_or_else(|_| "raw".to_string());
    let user_prompt = std::env::var("HIPFIRE_SMOKE_PROMPT").unwrap_or_else(|_| "Hello".to_string());
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
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

    eprintln!("\n=== forward_scratch prefill (paged) ===");
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
    assert!(n_nan == 0, "NaN in logits — paged forward path produced NaN");
    assert!(n_inf == 0, "Inf in logits — paged forward path produced Inf");

    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    indexed.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed[..5].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  top-5 token ids: {:?}", &indexed[..5]);
    let argmax = indexed[0].0;
    eprintln!("  argmax = {argmax}  (elapsed: {:?})", elapsed);

    if let Some(p) = &weights.pager {
        let p = p.borrow();
        eprintln!("Pager after prefill: resident {}, used {} bytes",
            p.resident_count(), p.vram_used_bytes());
    }

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
            let has_nan = l.iter().any(|v| v.is_nan() || v.is_infinite());
            assert!(!has_nan, "NaN/Inf at step {step}");
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

    if let Some(p) = &weights.pager {
        let p = p.borrow();
        eprintln!("\nPager final: resident {}, used {} bytes (catalog {})",
            p.resident_count(), p.vram_used_bytes(), p.registered_count());
    }
    eprintln!("\n=== PAGED SMOKE TEST PASSED ===");
}
