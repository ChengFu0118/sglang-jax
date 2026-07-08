# gpt-oss-120b Serving Sweep — SGLang-JAX on TPU v7x (post Fix #1)

ISL/OSL × batch-size scan of **`openai/gpt-oss-120b`** on a single **TPU v7x-8** (Ironwood) host,
SGLang-JAX, in the [`docs/performance/qwen3_benchmark.md`](../../docs/performance/qwen3_benchmark.md)
format. This is the **post-[Fix #1](gptoss_120b_fix1_layout.md)** result (expert weights pinned to
the `{2,1,0}` layout — the per-step relayout copy eliminated). **bf16 experts, bf16 KV.**

vLLM columns are omitted: bf16 gpt-oss-120b (~234 GB) does not fit vLLM-TPU's 1-chip-per-replica
limit (TP capped at 2 for gpt-oss), so a bf16-vs-bf16 120b comparison isn't runnable on vLLM without
a sharding change — see [`gptoss_120b_benchmark.md`](gptoss_120b_benchmark.md) for the MXFP4 vLLM
head-to-head.

## Test configuration

- **Test date**: 2026-07-08
- **Hardware**: one `tpu7x-standard-4t` node — topology `2x2x1` = 4 chips = **8 TPU cores**. Single host.
- **Model**: `openai/gpt-oss-120b` (117B total / 5.1B active, 128 experts, MXFP4 checkpoint,
  dequantized to **bf16** at load).
- **Code**: fork `gpt-oss-support` @ `52a818df` (Fix #1 layout). **A fresh
  `JAX_COMPILATION_CACHE_DIR` is required after the fix** — pre-fix executables expect the old
  `{0,2,1}` expert layout and crash with a layout-mismatch.
- **Client**: `sgl_jax.bench_serving` — `--dataset-name random --random-range-ratio 1
  --warmup-requests 0`, `ignore_eos` on (fixed output length), `--num-prompts = 3 × concurrency`,
  `request_rate = inf`.
- **Sweep**: ISL {1024, 4096, 8192} × OSL {1, 1024} × concurrency {8, 16, 32, 64, 128, 256}.
- **Methodology (matches qwen3 doc)**: **TTFT** is measured from the `OSL=1` runs; **ITL** and the
  throughputs from the `OSL=1024` runs. All reported values are `bench_serving` **medians** (the
  first request per shape pays untuned-v7x compile; the median is clean).

### Server startup

```bash
JAX_COMPILATION_CACHE_DIR=/workspace/jit_cache_benchfix \
PYTHONPATH=/workspace/repo/python \
python -m sgl_jax.launch_server \
  --model-path openai/gpt-oss-120b \
  --device tpu --tp-size 8 --dtype bfloat16 \
  --context-length 10240 --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 --page-size 128 \
  --max-running-requests 256 \
  --disable-radix-cache --skip-server-warmup --disable-precompile \
  --watchdog-timeout 3600 --host 0.0.0.0 --port 30011
```

## Detailed performance data (SGLang-JAX, bf16, post Fix #1)

Raw per-run JSONL: [`gptoss_120b_bench_data/sglang/gptoss120b_fix1_sweep.jsonl`](gptoss_120b_bench_data/sglang/gptoss120b_fix1_sweep.jsonl);
parsed CSV: [`gptoss_120b_fix1_sweep.csv`](gptoss_120b_fix1_sweep.csv).

| ISL/OSL | Batch Size | TTFT(ms) | ITL(ms) | Input_Throughput(tok/s) | Output_Throughput(tok/s) |
|---|---|---|---|---|---|
| 1024/1024 | 8   | 303.18   | 8.64  | 896.06    | 896.06  |
| 1024/1024 | 16  | 596.25   | 11.27 | 828.93    | 828.93  |
| 1024/1024 | 32  | 1200.94  | 14.70 | 1446.50   | 1446.50 |
| 1024/1024 | 64  | 2404.94  | 22.90 | 1813.30   | 1813.30 |
| 1024/1024 | 128 | 4810.25  | 28.21 | 3272.86   | 3272.86 |
| 1024/1024 | 256 | 9620.43  | 47.64 | 4028.00   | 4028.00 |
| 4096/1024 | 8   | 1166.23  | 8.13  | 3417.73   | 854.43  |
| 4096/1024 | 16  | 2341.97  | 10.91 | 4840.28   | 1210.07 |
| 4096/1024 | 32  | 4700.78  | 15.49 | 6378.73   | 1594.68 |
| 4096/1024 | 64  | 9380.89  | 22.88 | 7989.40   | 1997.35 |
| 4096/1024 | 128 | 18751.12 | 32.36 | 10099.45  | 2524.86 |
| 4096/1024 | 256 | 37443.96 | 47.70 | 12118.70  | 3029.68 |
| 8192/1024 | 8   | 2399.40  | 7.87  | 6179.96   | 772.50  |
| 8192/1024 | 16  | 4775.44  | 10.88 | 8207.14   | 1025.89 |
| 8192/1024 | 32  | 9589.82  | 16.73 | 9856.09   | 1232.01 |
| 8192/1024 | 64  | 19144.23 | 22.88 | 12316.77  | 1539.60 |
| 8192/1024 | 128 | 38177.31 | 36.37 | 13883.98  | 1735.50 |
| 8192/1024 | 256 | 76243.64 | 47.60 | 16729.88  | 2091.23 |

## Before/after Fix #1 (overlapping points)

The pre-fix [benchmark](gptoss_120b_benchmark.md) recorded two ISL 8192 / OSL 1024 points with the
identical server config (minus the layout fix). Fix #1 improves both decode (ITL / output tput) and
prefill (input tput) — the eliminated relayout copy ran in *every* forward, including each
chunked-prefill step:

| ISL/OSL | Batch | ITL pre → post (ms) | Output tok/s pre → post | Input tok/s pre → post |
|---|---|---|---|---|
| 8192/1024 | 16 | 49.02 → **10.88** (4.5× lower) | 216.3 → **1025.9** (4.7×) | 1730.5 → **8207.1** (4.7×) |
| 8192/1024 | 64 | 60.83 → **22.88** (2.7× lower) | 625.3 → **1539.6** (2.5×) | 5002.4 → **12316.8** (2.5×) |

## InferenceX interactivity plot

Throughput-per-chip vs interactivity (`1000/TPOT`) frontier vs NVIDIA B200/GB200 and vLLM-TPU
(both FP4): [`plots/sgljax_v7x_interactivity.png`](plots/sgljax_v7x_interactivity.png) (log-log) and
[`plots/sgljax_v7x_interactivity_zoom.png`](plots/sgljax_v7x_interactivity_zoom.png) (TPU-region
zoom). Per-chip peak: **~1007 tok/s/chip** (1k/1k) and **~523** (8k/1k) — see
[`plots/README.md`](plots/README.md).

## Observations

- **ITL is roughly context-independent at fixed batch** (e.g. at concurrency 256, ITL ≈ 47.6 ms for
  ISL 1024/4096/8192): decode step time tracks batch size, not context length.
- **Output throughput scales with concurrency** (ISL 1024: 896 → 4028 tok/s from bs 8 → 256) and
  **decreases with longer context** at fixed concurrency (bs 256: 4028 → 3030 → 2091 for ISL
  1024/4096/8192) as each decode step reads more KV.
- **Input throughput rises with ISL and concurrency**, peaking at **16.7k tok/s** (ISL 8192, c=256):
  longer prefills fill the chunked-prefill pipeline better.
- **TTFT grows ~linearly with concurrency** (rate=inf, so all requests queue against the prefill).
- Peak aggregate output throughput here is **4028 tok/s** (ISL 1024, c=256); the aggregate includes
  prefill/queueing time in the denominator, so it sits below the steady-state decode ceiling.

## Caveats

- **bf16 experts + bf16 KV, TP=8** (SGLang-JAX's config); head_dim 64 padded to 128, untuned RPA v3
  block sizes on v7x, `force_gmm_v1`. The remaining ranked fixes (tuned hd64 kernel, fp8 KV,
  `gmm_v2`, native MXFP4) are unchanged — see [`gptoss_120b_bottleneck_analysis.md`](gptoss_120b_bottleneck_analysis.md).
- The before/after table compares against a prior partial run; only those two points overlap.

## Benchmark script

```bash
#!/bin/bash
input_seq_lens=(1024 4096 8192); output_seq_lens=(1 1024)
concurrencies=(8 16 32 64 128 256); npc=3
for isl in "${input_seq_lens[@]}"; do
  for osl in "${output_seq_lens[@]}"; do
    for c in "${concurrencies[@]}"; do
      python -m sgl_jax.bench_serving --backend sgl-jax --host 0.0.0.0 --port 30011 \
        --dataset-name random --num-prompts $((npc*c)) \
        --random-input-len $isl --random-output-len $osl \
        --max-concurrency $c --random-range-ratio 1 --warmup-requests 0 \
        --output-file bench_results.jsonl
    done
  done
done
```
