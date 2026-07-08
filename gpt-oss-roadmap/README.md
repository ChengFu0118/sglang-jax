# GPT-OSS on SGLang-JAX — Roadmap & Progress

Status of bringing OpenAI **gpt-oss** (`gpt-oss-20b`, `gpt-oss-120b`) to SGLang-JAX on TPU.

## TL;DR

- ✅ **gpt-oss-20b and -120b run on TPU v7x (Ironwood) in `--dtype bfloat16`** and produce
  correct output ("Paris", "4", "the quick brown fox … the lazy dog"). The full model path is
  implemented and validated: MXFP4 weight load, attention sinks, alternating sliding/full
  attention, YaRN RoPE, QKV/O bias, and the clamped `swigluoai` MoE with per-expert biases.
- ✅ **bf16 NaN FIXED.** Root cause: the megablox **`gmm_v2`** grouped-matmul kernel returns
  NaN in bf16 under the heavily-imbalanced group sizes that token padding creates (padding
  tokens route to one expert). Fix: `force_gmm_v1` for gpt-oss (numerically correct v1 kernel).
  Details below. fp32 also still works.
- ✅ **gpt-oss-120b head-to-head vs vLLM `tpu-inference` on v7x-8** — see
  [`gptoss_120b_benchmark.md`](gptoss_120b_benchmark.md). vLLM leads ~3.6–6.7× on throughput
  (v7x-tuned `hd64` kernels + fp8 KV + DP=4 + MXFP4 4-bit experts vs SGLang-JAX's untuned RPA +
  `gmm_v1` + bf16 KV + bf16 experts); the gap tracks kernel/config differences, not the model port.
- 🔬 **Profile-backed bottleneck analysis** — see
  [`gptoss_120b_bottleneck_analysis.md`](gptoss_120b_bottleneck_analysis.md). The #1 fixable
  bottleneck is a per-step **expert-weight relayout copy (~35% of decode time)**; then the 4× KV
  (head_dim pad + bf16-vs-fp8) and the TP=8 collective tax. Includes a Qwen3-MoE control proving
  SGLang-JAX's MoE path is healthy, and a ranked fix roadmap.

## What was implemented (this session)

| File | Change |
|---|---|
| `python/sgl_jax/srt/models/gpt_oss.py` | **New.** `GptOssForCausalLM` (`EntryClass`) + custom `load_weights`. |
| `python/sgl_jax/srt/utils/quantization/mxfp4.py` | **New.** `dequantize_mxfp4` (+ `u8_unpack_e2m1`, `e8m0_to_fp32`). |
| `python/sgl_jax/test/test_mxfp4_gpt_oss.py` | **New.** MXFP4 round-trip + `swigluoai` unit tests (wired into `unit-test-cpu`). |
| `python/sgl_jax/srt/layers/moe.py` | EPMoE gains `activation="swigluoai"` + `use_expert_bias`. |
| `python/sgl_jax/srt/server_args.py`, `.../model_runner_kv_cache_mixin.py` | allow `--kv-cache-dtype fp32`. |

Key design points (all reuse existing primitives):

- **Experts (MXFP4 → bf16 at load).** gpt-oss ships experts as MXFP4 (4-bit `e2m1` codes +
  `e8m0` group-32 scales). The loader dequantizes to bf16, splits the **fused, interleaved**
  `gate_up` (gate = `[:, ::2]`, up = `[:, 1::2]`), transposes to EPMoE `[E, k, n]`, and shards
  onto the EPMoE `moe_mesh`. Non-quantized siblings (attn, router, embed, lm_head) load directly.
- **`swigluoai` activation.** Clamped SwiGLU with `(up + 1)` (α = 1.702, `limit = swiglu_limit`).
- **Per-expert bias via `gmm(rhs_bias=…)`.** The down-proj (`wo`) bias is pre-scaled by
  `1/tp_size`: EPMoE `psum`-reduces the `wo` output across the `tensor` axis, so a per-shard
  `rhs_bias` would otherwise be added `tp_size` times. Gate/up biases live on the (unreduced)
  sharded intermediate dim and are added once. Verified against a NumPy reference.
- **Router.** gpt-oss softmaxes **after** top-k (not over the full logit vector): raw logits +
  router bias → `top_k` → `softmax` over the selected experts.
- **Attention.** Sinks (per-head, RPA v3 / FlashAttention `attention_sink=`), per-layer sliding
  window from `config.layer_types` (128-token window on `sliding_attention` layers, full
  otherwise), YaRN RoPE via `get_rope("yarn")`, and `head_dim=64` padded to 128 for the kernel.
- **Config.** No custom config class — stock `transformers>=4.55` owns `gpt_oss`. sglang treats
  `quant_method="mxfp4"` as unsupported → `quantization_config=None`, so EPMoE stays bf16 and we
  dequantize the experts ourselves.

## How to run (bf16, single host)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path openai/gpt-oss-20b \
  --trust-remote-code \
  --tp-size 8 --device tpu \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 30000
```

`--tp-size` = total JAX devices (v7x exposes 2 devices/chip, so a 4-chip v7x-8 host is `--tp-size 8`).
`gpt-oss-120b` runs the identical path (`--model-path openai/gpt-oss-120b`) on v7x-8 (fits in HBM).
bf16 is now correct (see the bf16 fix below); fp32 also works via `--dtype float32`.

## Verification done

- 8/8 unit tests pass on TPU (`python -m unittest test_mxfp4_gpt_oss`): MXFP4 dequant round-trip,
  `swigluoai` equivalence vs the v2 kernel, e8m0/interleave checks.
- Full 24-layer gpt-oss-20b in fp32 on v7x-8 → coherent completions:
  - `"The capital of France is"` → `" Paris."`
  - `"Q: What is 2+2?\nA:"` → `" 4"` (+ coherent Q&A)
  - `"The quick brown fox"` → `" jumps over the lazy dog."`

## bf16 → NaN — RESOLVED (megablox gmm_v2 kernel)

**Root cause:** the megablox **`gmm_v2`** grouped-matmul kernel returns NaN for gpt-oss's bf16
experts when a padded (short) prompt routes **all its padding tokens to a single expert**,
producing a heavily-imbalanced `group_sizes`. In serving, prompts are always padded up to a
compile bucket, so this fires constantly. `gmm_v1` is numerically correct for the same inputs;
full fp32 also masks it (its routing avoids the pathological grouping).

**Fix:** `gmm(force_v1=...)` in `kernels/gmm/megablox_gmm_backend.py`, threaded through
`EPMoE(force_gmm_v1=...)` in `layers/moe.py`; gpt-oss sets `force_gmm_v1=True` in
`models/gpt_oss.py`. Verified: gpt-oss-20b **and** -120b bf16 on v7x-8 → coherent, no NaN;
the 8 mxfp4 unit tests still pass.

**How it was localized** (the prior "instrumentation wall" was the *server's stale
`JAX_COMPILATION_CACHE_DIR`*, not a real wall — in a fresh single-process run `jax.debug.print`
works fine):

- Built a fast single-process compiled-forward probe (`sgl_jax.dbg_forward` reusing
  `bench_one_batch` helpers; `DBG_TOKEN_BUCKET=64` reproduces the padding trigger). 2-layer bf16
  20b, 5-token prompt padded to a 64-bucket → NaN; unpadded → finite.
- Per-layer probes: first NaN appears at **L0 MoE output** (`experts_out`) — attention is clean,
  and the MoE inputs (router logits, top-k weights) are finite. Exactly the 5 real-token rows go
  NaN.
- A/B inside the MoE: fp32 gmm compute → still NaN; `zero_initialize=True` → still NaN; disable
  expert bias → still NaN; **force `gmm_v1` → finite**. Full fp32 model → finite. So the bug is
  the bf16 `gmm_v2` path under this group structure (Qwen3-MoE is unaffected — no expert bias /
  no such padding stress in its tested paths).

**Follow-up (out of scope):** a proper `gmm_v2` fix would let gpt-oss use the faster v2 kernel;
until then `force_gmm_v1` is a documented throughput caveat (see the benchmark writeup).

## Roadmap / follow-ups

1. ✅ **Fix bf16 serving** (compiled-path NaN) — done (`force_gmm_v1`; see above).
2. ✅ **Head-to-head benchmark** vs vLLM `tpu-inference` on v7x — done, see
   [`gptoss_120b_benchmark.md`](gptoss_120b_benchmark.md).
3. ✅ **gpt-oss-120b** on v7x-8 (`--tp-size 8`) — done, bf16, coherent + benchmarked.
4. **Fix megablox `gmm_v2` bf16** for imbalanced group_sizes → drop `force_gmm_v1` and recover
   MoE throughput.
5. **v7x-tuned RPA/gmm block sizes** for gpt-oss shapes (avoid the `LOOKUP MISS` heuristic) and a
   dedicated `head_dim=64` attention path (vLLM ships `ragged_paged_attention_hd64`).
6. **Native MXFP4 grouped-matmul** (keep experts 4-bit) for HBM/throughput.
7. **fp8 KV cache** for gpt-oss (vLLM uses it; halves KV bandwidth/footprint).
8. **EPLB / expert-parallel tuning** for 128 experts (120b).

See `sglang_jax_vs_tpu_inference.md` (same dir) for the model-coverage matrix and the benchmark
methodology reference.
