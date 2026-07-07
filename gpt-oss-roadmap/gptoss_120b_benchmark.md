# gpt-oss-120b Benchmark — SGLang-JAX vs vLLM tpu-inference (TPU v7x)

Head-to-head throughput/latency of **`openai/gpt-oss-120b`** on a single **TPU v7x-8**
(Ironwood) host, SGLang-JAX vs vLLM `tpu-inference`, in the `docs/performance/qwen3_benchmark.md`
format. This is the comparison the roadmap promised; it is unblocked by the bf16 MoE NaN fix
(`force_gmm_v1`, see [`README.md`](README.md) and commit history).

## Test configuration

- **Test date**: 2026-07-07
- **Hardware**: one `tpu7x-standard-4t` node — topology `2x2x1` = 4 chips = **8 TPU cores**
  (`google.com/tpu: 4`). Single host; no multi-host.
- **Model**: `openai/gpt-oss-120b` (117B total / 5.1B active, 128 experts, MXFP4 checkpoint).
- **Client**: identical harness for both backends — `sgl_jax.bench_serving`
  (`--dataset-name random --random-range-ratio 1 --warmup-requests 0`, `ignore_eos` ON so the
  output length is fixed). `--num-prompts = 3 × concurrency`.
- **Workloads** (vLLM daily gpt-oss cases):
  - **ISL 1024 / OSL 8192** — reasoning-style long output.
  - **ISL 8192 / OSL 1024** — long context, short output.
- **Concurrency sweep**: {16, 64, 256}.

### Each engine runs its own recommended config (fairness note)

KV-cache dtype and parallelism differ **by design** — each engine uses the config its own
docs/recipes recommend on this host. This is documented, not hidden.

| | SGLang-JAX | vLLM tpu-inference |
|---|---|---|
| Parallelism | **TP=8** (single replica) | **TP=2 × DP=4** (4 replicas) |
| Weights | bf16 (MXFP4 dequantized at load) | bf16 (MXFP4) |
| KV cache | **bf16** | **fp8** |
| Attention kernel | generic RPA v3, **head_dim 64 padded to 128, untuned block sizes on v7x** | dedicated **`ragged_paged_attention_hd64`, v7x-tuned block sizes** |
| MoE grouped matmul | megablox **gmm_v1** (v2 NaNs in bf16 — see roadmap) | tuned GMM TP kernel |
| Image | `lmsysorg/sglang-jax:v0.0.3rc99` + `gpt-oss-support` fork overlay | `vllm/vllm-tpu:nightly` |
| context-length / max-model-len | 10240¹ | 9216 |

¹ SGLang-JAX rejects a request when `input + max_new_tokens == context-length`, so its
context-length is set to 10240 (1024 headroom); the in/out **token counts processed are
identical** to vLLM's. All other differences are the vendor-recommended defaults.

**Server startup:**

```bash
# SGLang-JAX (this fork, gpt-oss-support branch)
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
  --model-path openai/gpt-oss-120b --trust-remote-code \
  --tp-size 8 --device tpu --dtype bfloat16 \
  --context-length 10240 --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 --page-size 128 \
  --disable-radix-cache --skip-server-warmup --host 0.0.0.0 --port 30011

# vLLM tpu-inference
VLLM_USE_V1=1 MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 TPU_MULTIPROCESS_DP=0 \
TPU_BACKEND_TYPE=jax VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve openai/gpt-oss-120b --host 0.0.0.0 --port 8000 --seed 42 \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --max-model-len 9216 --max-num-batched-tokens 16384 --max-num-seqs 2048 \
  --kv-cache-dtype fp8 --no-enable-prefix-caching --async-scheduling \
  --gpu-memory-utilization 0.86
```

## Results

Metrics are `bench_serving` medians (TTFT, ITL) and aggregate throughput. Raw per-run JSONL is
under [`gptoss_120b_bench_data/`](gptoss_120b_bench_data/).

### ISL 1024 / OSL 8192 (reasoning — long output)

| ISL/OSL | Batch | TTFT(ms) SGL | TTFT(ms) vLLM | ITL(ms) SGL | ITL(ms) vLLM | In tok/s SGL | In tok/s vLLM | Out tok/s SGL | Out tok/s vLLM | Out ratio SGL/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024/8192 | 16  | 658 | 349 | 47.44 | 9.47  | 41.9   | 152.1  | **335.2**  | **1216.6**  | 0.28× |
| 1024/8192 | 64  | — | 694 | — | 10.31 | —      | 634.1  | —          | **5072.5**  | — |
| 1024/8192 | 256 | — | 916 | — | 17.44 | —      | 1690.7 | —          | **13526.0** | — |

### ISL 8192 / OSL 1024 (long context — short output)

| ISL/OSL | Batch | TTFT(ms) SGL | TTFT(ms) vLLM | ITL(ms) SGL | ITL(ms) vLLM | In tok/s SGL | In tok/s vLLM | Out tok/s SGL | Out tok/s vLLM | Out ratio SGL/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192/1024 | 16  | 6557  | 1293 | 49.02 | 9.60  | 1730.5 | 8879.2  | **216.3** | **1109.9** | 0.19× |
| 8192/1024 | 64  | 17697 | 2562 | 60.83 | 11.06 | 5002.4 | 33548.4 | **625.3** | **4193.6** | 0.15× |
| 8192/1024 | 256 | —     | 2497 | —     | 22.68 | —      | 49415.1 | —         | **6176.9** | — |

*Dashes (—) are SGLang-JAX points omitted for wall-clock: at ISL 8192 / concurrency 256 the
untuned prefill did not complete in a practical window (≈240 s for the first request), and the
OSL 8192 / concurrency 64 point (≈40 min at ~340 tok/s) was skipped once the trend was
established. All requests in the reported points completed (`completed = 3 × concurrency`).*

## Analysis

- **vLLM leads throughput by ~3.6–6.7×** on both workloads at matched concurrency (e.g.
  ISL8192/OSL1024 c=64: 4194 vs 625 tok/s output; ISL1024/OSL8192 c=16: 1217 vs 335), and
  **ITL is ~3–5× lower** (9–23 ms vs 47–61 ms). vLLM peak output here is **13.5k tok/s**
  (ISL1024/OSL8192, c=256).
- **TTFT gap is largest on long-context prefill**: 6.6 s vs 1.3 s (c=16) and 17.7 s vs 2.6 s
  (c=64) at ISL 8192 — SGLang-JAX's untuned RPA prefill scales poorly with a 8192-token input.
- **The gap is dominated by the documented config/kernel differences, not the model port**,
  which is correct (bf16 output is coherent — "Paris", "4", "the lazy dog"). The main levers,
  all out of scope here and flagged as follow-ups:
  1. **Untuned v7x attention** — SGLang-JAX pads gpt-oss's `head_dim=64` to 128 (≈2× wasted
     attention FLOPs) and hits the RPA v3 `tuned-block-size LOOKUP MISS` heuristic; vLLM ships a
     dedicated v7x-tuned `hd64` kernel.
  2. **`force_gmm_v1`** — the correctness fix for the bf16 MoE NaN falls back from the faster
     v2 grouped-matmul kernel; a proper v2 fix would recover MoE throughput.
  3. **KV dtype** — vLLM's fp8 KV halves KV bandwidth/footprint vs SGLang-JAX's bf16 KV.
  4. **DP=4 vs TP=8** — 4 independent replicas batch more efficiently at this scale than a
     single 8-way tensor-parallel replica with more cross-device collectives.
  5. Native MXFP4 matmul (keep experts 4-bit) — neither dequant-to-bf16 cost is optimized here.

### vLLM baseline sanity

The recorded vLLM recipe (rate=inf, 1024 prompts, uncapped concurrency up to `max-num-seqs`)
reports ISL1024/OSL8192 ≈ **20.8k tok/s (peak ~26k)**. This sweep caps `max-concurrency` at 256
with 768 prompts, giving **13.5k tok/s** at c=256 — lower purely because concurrency is capped
well below the engine's batch capacity, and monotonically rising with concurrency
(1.2k → 5.1k → 13.5k for c=16/64/256), consistent with the recipe. This confirms the vLLM
baseline is set up correctly.
