# GPT-OSS on SGLang-JAX — Roadmap & Progress

Status of bringing OpenAI **gpt-oss** (`gpt-oss-20b`, `gpt-oss-120b`) to SGLang-JAX on TPU.

## TL;DR

- ✅ **gpt-oss-20b runs on TPU v7x (Ironwood) and produces correct output in `--dtype float32`.**
  The full model path is implemented and validated: MXFP4 weight load, attention sinks,
  alternating sliding/full attention, YaRN RoPE, QKV/O bias, and the clamped `swigluoai` MoE
  with per-expert projection biases.
- ⚠️ **bf16 serving currently NaNs** in the compiled forward (the eager forward and fp32 are
  correct). This is the one open blocker — details below. **Serve with `--dtype float32` for now.**
- ⏭️ `gpt-oss-120b` (v7x-8, `--tp-size 8`) and the head-to-head throughput benchmark vs
  vLLM `tpu-inference` are gated on the bf16 fix (a fair bf16-vs-bf16/MXFP4 comparison).

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

## How to run (fp32, single host)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path openai/gpt-oss-20b \
  --trust-remote-code \
  --tp-size 8 --device tpu \
  --dtype float32 \
  --host 0.0.0.0 --port 30000
```

`--tp-size` = total JAX devices (v7x exposes 2 devices/chip, so a 4-chip v7x-8 host is `--tp-size 8`).
`gpt-oss-120b` should run the identical path at fp32 on v7x-8 (fits in HBM).

## Verification done

- 8/8 unit tests pass on TPU (`python -m unittest test_mxfp4_gpt_oss`): MXFP4 dequant round-trip,
  `swigluoai` equivalence vs the v2 kernel, e8m0/interleave checks.
- Full 24-layer gpt-oss-20b in fp32 on v7x-8 → coherent completions:
  - `"The capital of France is"` → `" Paris."`
  - `"Q: What is 2+2?\nA:"` → `" 4"` (+ coherent Q&A)
  - `"The quick brown fox"` → `" jumps over the lazy dog."`

## Known issue: bf16 → NaN (open)

In bf16 the server emits NaN logits (all `"!"`); **fp32 is correct**. The model math is fine —
the NaN is specific to the **compiled/paged serving forward**:

- Full **fp32 24-layer server → coherent**.
- **Offline eager forward** (real weights, bf16, even with token padding) → **finite** (2-layer
  final amax ≈ 232, i.e. not even a massive-activation regime).
- Server bf16: **1 layer fine, ≥2 layers NaN** (any layer type: 2×sliding, 2×full both NaN).

Ruled out (each a full recompile+test on v7x):

- Attention — fp32 q/k/v **and** fp32 KV cache → still NaN.
- MoE — forced EPMoE to fp32 → still NaN.
- **fp32 activations + bf16 weights** (LinearBase outputs, MoE, KV, attention all fp32) →
  **still NaN**, yet full fp32 works. Adding fp32 did **not** monotonically help, so the trigger
  tracks the model **`dtype` flag itself** (a dtype-conditional *structural* path — e.g. KV-cache
  packing bf16=2 vs fp32=1, or compilation bucketing — not any single op's precision).
- Cross-layer XLA fusion (`optimization_barrier`), head_dim<128 padding, attention sinks,
  `softmax_dtype=fp32`, padded-token masking in the MoE → none fixed it.

Localization is blocked by tooling: sglang's executor drops all host callbacks
(`jax.debug.print`, `jax.pure_callback` with a keep-alive, file-write callbacks all silently
DCE'd), `JAX_DEBUG_NANS=1` hangs (incompatible with the Pallas/compiled path), and Pallas requires
jit so the server forward can't run eagerly. Note: vLLM `tpu-inference` ships a **dedicated
`ragged_paged_attention_hd64` kernel** for gpt-oss rather than reusing the generic bf16 path —
consistent with a bf16-specific compiled-path issue here.

**Next lever:** reproduce the real `ModelRunner` compiled forward in a script and bisect by
forcing fp32 per dtype-conditional structural choice (KV packing / bucketing), or get an
sglang-jax maintainer / XLA HLO dump to find the unstable fused op.

## Roadmap / follow-ups

1. **Fix bf16 serving** (compiled-path NaN) — unblocks the fair benchmark and 120b memory budget.
2. **Head-to-head benchmark** vs vLLM `tpu-inference` on v7x, in the `qwen3_benchmark.md` format
   (unified `sgl_jax.bench_serving` client, concurrency sweep, TTFT/ITL/throughput).
3. **gpt-oss-120b** on v7x-8 (`--tp-size 8`) — same code path, first throughput point.
4. **Native MXFP4 grouped-matmul** (keep experts 4-bit) for HBM/throughput.
5. **v7x-tuned RPA/gmm block sizes** for gpt-oss shapes (avoid the `LOOKUP MISS` heuristic).
6. **EPLB / expert-parallel tuning** for 128 experts (120b).

See `sglang_jax_vs_tpu_inference.md` (same dir) for the model-coverage matrix and the benchmark
methodology reference.
