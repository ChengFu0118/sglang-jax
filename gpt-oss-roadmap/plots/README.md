# InferenceX interactivity plots — gpt-oss-120b, SGLang-JAX TPU v7x

Throughput-per-accelerator vs interactivity (InferenceX style) for `openai/gpt-oss-120b`,
featuring **SGLang-JAX on TPU v7x (bf16, post [Fix #1](../gptoss_120b_fix1_layout.md))** against
NVIDIA B200/GB200 and vLLM-TPU v7x (both FP4).

- **`sgljax_v7x_interactivity.png`** — log-log, full comparison (all series).
- **`sgljax_v7x_interactivity_zoom.png`** — linear, zoomed to the TPU operating region so the
  SGLang-JAX and vLLM-TPU points are legible (NVIDIA curves extend above the frame).
- **`plot_sgljax_interactivity.py`** — regenerates both from the sweep JSONL + NVIDIA data.

## Axes & conventions (InferenceX)

- **x = interactivity** = output tok/s per user = `1000 / TPOT_ms`.
- **y = output throughput per accelerator** = aggregate output tok/s / **accelerators**.
- **Per-accelerator = per chip.** A v7x-8 node = **4 chips** (2 cores/chip). SGLang-JAX runs
  **TP=8** (one replica across all 4 chips); vLLM runs **TP=2×DP=4** (4 one-chip replicas) — both
  divide the node's aggregate by 4, so they're directly comparable per chip.
- Each curve is traced by the **concurrency** sweep {8,16,32,64,128,256}; high concurrency → high
  throughput / low interactivity (lower-right → upper-left tradeoff).

## Data sources

- **SGLang-JAX v7x (bf16):** this repo's sweep — [`../gptoss_120b_bench_data/sglang/gptoss120b_fix1_sweep.jsonl`](../gptoss_120b_bench_data/sglang/gptoss120b_fix1_sweep.jsonl)
  (see [`../gptoss_120b_sweep.md`](../gptoss_120b_sweep.md)). `interactivity = 1000/median_tpot_ms`,
  `tput/chip = output_throughput / 4`.
- **NVIDIA B200/GB200 (FP4):** SemiAnalysis InferenceX DB snapshot 2026-06-29 (`/tmp/ix_gptoss_points.json`),
  upper-envelope Pareto frontier per workload.
- **vLLM-TPU v7x (FP4):** reference points from the tpu-inference steady-state study (per-chip).

## Takeaways (per chip)

| workload | SGLang-JAX v7x bf16 (this work) | vLLM-TPU v7x FP4 | NVIDIA B200 FP4 (peak) |
|---|---|---|---|
| 1k/1k | ~1007 tok/s/chip @ 18 tok/s/user | ~2600 tok/s/chip | ~9.8k–14.7k tok/s/GPU |
| 8k/1k | ~523 tok/s/chip @ 12 tok/s/user | ~1600 tok/s/chip | ~5.7k–7.3k tok/s/GPU |

SGLang-JAX bf16 trails vLLM-TPU FP4 by ~2.5× per chip (FP4 experts + tuned hd64 kernel + fp8 KV +
DP batching) and NVIDIA FP4 by ~10× per accelerator (larger/newer chip, higher concurrency range).
Fix #1 (the expert-layout relayout removal) is what lifted the SGLang-JAX frontier ~3–4×; the
remaining gap is the ranked follow-ups (native MXFP4, fp8 KV + hd64, DP/EP, gmm_v2) in
[`../gptoss_120b_bottleneck_analysis.md`](../gptoss_120b_bottleneck_analysis.md).

*Note: at 1k/1k the c16 point is slightly Pareto-dominated by c8 (measurement noise at low
concurrency, small prompt count), so the frontier line skips it.*
