# Fix #1 — expert-weight row-major layout (SGLang-JAX gpt-oss-120b, TPU v7x)

Result of **Fix #1** from the [bottleneck analysis](gptoss_120b_bottleneck_analysis.md): store the
MoE expert weights on-device in the layout the megablox `gmm` kernel tiles for, eliminating the
per-step **expert-weight relayout copy** that dominated decode. Verified on v7x-8.

**Status: DONE + verified.** Commit `52a818df` (`perf(gpt-oss): pin expert weights row-major to
kill per-step relayout copy`) on branch `gpt-oss-support`. **bf16 experts** throughout (this is the
dequantized bf16 MoE path, not native MXFP4).

## The change

`python/sgl_jax/srt/models/gpt_oss.py` only (23 insertions, 8 deletions):

- `_assign` gains an opt-in `row_major` flag → builds
  `Format(Layout(major_to_minor=(0..n-1)), sharding)` and uses it in **both** placement paths
  (real `jax.make_array_from_callback` and dummy `jit out_shardings`).
- `_load_experts` passes `row_major=True` for the three gmm weights `wi_0` / `wi_1` / `wo`
  (real + dummy branches). Norms / biases / attention / router are unchanged.

`major_to_minor=(0,1,2)` for the 3-D `[E,k,n]` expert weight == XLA layout `{2,1,0}` (n / last-dim
contiguous), which is exactly what the megablox `gmm` kernel reads. Previously the weights lived on
device as `{0,2,1}`, so XLA inserted a layout-conversion copy of **every** expert weight
(`wi_0`/`wi_1`/`wo` × 36 layers ≈ 108 copies, ~530 MB each) **inside every forward step** — the same
static weights reshuffled once per decode step. Mirrors the tpu-inference `general_device_put` /
`moe_weights.py` layout pattern. **Layout is physical byte order only — values (numerics) are
unchanged.**

## Profiling method

`bench_one_batch --profile --load-format dummy` on `gpt-oss-120b`, **bs=64 / in=1024 / out=64,
bf16, tp=8, page-size 128**, on the v7x-8 pod `gptoss-sgl` (cloud-devkit-gke). Device time summed
over the 8 TPU cores from the xprof Chrome trace (`/device:TPU:* / XLA Ops`), bucketed by op into
categories. Baseline and fix use the identical command (dummy weights — the relayout is
value-independent). Same methodology as the bottleneck analysis. Raw traces:
`prof_out/hs64_trace.json.gz` (baseline) vs `prof_out/fixv1_trace.json.gz` (fix); parsed numbers in
[`gptoss_120b_fix1_profile.csv`](gptoss_120b_fix1_profile.csv) and
[`gptoss_120b_fix1_throughput.csv`](gptoss_120b_fix1_throughput.csv).

## Result — decode device time by category

| Category | Baseline ms | Baseline % | Fix ms | Fix % | Δ ms |
|---|---|---|---|---|---|
| **memory/layout** (copies) | **5817.3** | **36.1%** | **1570.3** | **13.1%** | **−73.0%** |
| matmul/fusion | 5138.5 | 31.9% | 5165.4 | 43.1% | +0.5% |
| collective (TP=8 psum/all-reduce) | 2720.3 | 16.9% | 2743.3 | 22.9% | +0.8% |
| moe/gmm (Pallas) | 1086.6 | 6.7% | 1098.1 | 9.2% | +1.1% |
| attention (RPA v3) | 902.2 | 5.6% | 943.1 | 7.9% | +4.5% |
| other | 444.1 | 2.8% | 461.9 | 3.9% | +4.0% |
| **Total device time** | **16109.0** | 100% | **11982.1** | 100% | **−25.6%** |

Every category is **unchanged in absolute ms** except memory/layout, which drops by **4247 ms**
(−73%) — exactly the eliminated expert relayout. The other categories' *percentages* rise only
because the denominator shrank. The `bf16[128,2880,360]{0,2,1}→{2,1,0}` expert-relayout copies
(~37 ms each in the baseline) are **gone** from the fix trace; the residual 13.1% memory/layout is
unrelated 2-D activation transposes (`[25136,2880]`, `[65536,2880]`), not expert weights.

## Result — decode throughput (bs=64)

| | Baseline bf16 | Fix (row-major) bf16 | Speedup |
|---|---|---|---|
| decode median, warm | **1183 tok/s** | **4045 tok/s** | **3.42×** |
| decode median, profiled | 1208 tok/s | 4378 tok/s | 3.62× |

The 3.4× (larger than the analysis's conservative "single-digit copy %" estimate) reflects that
decode is memory-bound: removing ~108 × 530 MB ≈ **57 GB/step** of pointless relayout traffic
dominates the step time.

## Numerics unchanged (real weights)

Real `gpt-oss-120b` bf16 served (`--dtype bfloat16 --context-length 10240`, tp=8) — coherent,
correct completions (temperature 0):

- `"The capital of France is"` → `" Paris."`
- `"2+2="` → `"4, 2+3=5"`
- `"The quick brown fox jumps over the"` → `" lazy dog."`

Unit tests `python -m unittest sgl_jax.test.test_mxfp4_gpt_oss` → **8/8 pass**.

## Scope & caveats

- **bf16 experts.** These numbers are the dequantized-bf16 MoE path. gpt-oss ships MXFP4; SGLang-JAX
  dequantizes to bf16 at load, so the experts are bf16 in HBM (4× the bytes of native 4-bit). Native
  MXFP4 experts (w4a16) remains a separate follow-up.
- **This fix is gpt-oss-only** (`gpt_oss.py`). The same latent relayout exists in the shared EPMoE
  loader (`weight_utils.py`) and affects Qwen3-MoE (~9%); generalizing there is a noted follow-up.
- Config still differs from vLLM (TP=8 vs TP=2×DP=4; bf16 KV vs fp8 KV; untuned RPA vs tuned hd64) —
  see [`gptoss_120b_benchmark.md`](gptoss_120b_benchmark.md). This fix addresses only the relayout
  copy bottleneck; the other ranked fixes are unchanged.
