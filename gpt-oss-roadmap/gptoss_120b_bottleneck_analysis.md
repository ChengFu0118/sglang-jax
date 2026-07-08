# gpt-oss-120b bottleneck analysis — SGLang-JAX vs vLLM tpu-inference (TPU v7x)

Why SGLang-JAX is ~3.6–6.7× behind vLLM `tpu-inference` on gpt-oss-120b (see the
head-to-head in [`gptoss_120b_benchmark.md`](gptoss_120b_benchmark.md)), what the actual
bottlenecks are, and how to fix them — backed by xprof profiles of the compiled decode
path on a v7x-8 host.

## TL;DR

1. **The head-to-head is *not* bf16-vs-bf16.** vLLM runs **MXFP4 (4-bit, w4a16) experts +
   fp8 KV**; SGLang-JAX runs **bf16 (dequantized) experts + bf16 KV**. (Correction to the
   benchmark doc's "both bf16 weights".)
2. **SGLang-JAX's MoE path is healthy** — the gap is *not* the MoE kernel. Proven by a
   Qwen3-MoE-30B control (see below).
3. **The single biggest, most fixable bottleneck is a per-step expert-weight *relayout copy*
   (~28–35% of decode device time)** — a pure on-device parameter-layout issue, not a kernel
   rewrite, not head_dim, not parallelism.
4. The **4× KV cache** (head_dim padding 2× + bf16-vs-fp8 2×) is the long-context / high-
   concurrency bottleneck and a hard *capacity* ceiling (it's why the in8192 c=256 point
   couldn't even run on SGLang-JAX).
5. The **TP=8 collective tax (~16%)** is a general, smaller lever vLLM avoids with TP=2×DP=4.

Ranked fixes: **(1) expert-weight layout → (2) DP/EP parallelism → (3) fp8 KV + native
head_dim=64 → (4) MXFP4 experts.**

## Configs compared (what actually differs)

| | SGLang-JAX (as benchmarked) | vLLM tpu-inference (recommended) |
|---|---|---|
| Parallelism | **TP=8, DP=1** (1 replica) | **TP=2 × DP=4** (4 replicas) |
| MoE experts | **bf16** (MXFP4 dequantized at load) | **MXFP4 4-bit (w4a16)**, GMM does 4-bit×bf16 |
| KV cache | **bf16**, head_dim padded 64→**128** | **fp8**, native head_dim **64** |
| Attention kernel | generic RPA v3 (pads head_dim to 128) | dedicated `ragged_paged_attention_hd64` |
| MoE kernel | megablox `gmm` (gpt-oss forced to `gmm_v1`) | tuned GMM TP kernel (supports expert bias) |

These are each engine's *recommended* config; the differences are by design and documented,
not a misconfiguration.

## Profiling method

`bench_one_batch --profile` (bs=64, in=1024, out=64, bf16), decode-dominant. Device time is
summed over the 8 TPU cores from the xprof Chrome trace (`/device:TPU:* / XLA Ops` rows). The
same methodology is applied to every model below for apples-to-apples category shares.

## Where SGLang-JAX spends decode time (gpt-oss-120b)

| Category | Device time | % |
|---|---|---|
| **memory/layout — `copy.*` (4,144 distinct)** | 5,815 ms | **34.6%** |
| matmul/fusion | 5,128 ms | 30.6% |
| **collective — `psum` / `all-reduce` (TP=8)** | 2,723 ms | 16.2% |
| moe/gmm (Pallas) | 1,758 ms | 10.5% |
| attention (RPA v3, Pallas) | 916 ms | 5.5% |
| other | 444 ms | 2.6% |

At serving decode batch sizes the useful matmuls are small and memory-bound; the run is
dominated by **layout copies (~35%)** and **TP=8 collectives (~16%)** — *not* the MoE or
attention kernels.

## Bottleneck #1 — per-step expert-weight relayout copy (~35%)

The dominant copy op:

```
copy  bf16[128,2880,360]{2,1,0}  ⟵  copy(bf16[128,2880,360]{0,2,1} %param.1105)
      bytes_accessed: 530 MB   hlo_category: "data formatting"
```

- A tensor's **layout** is which dim is contiguous in HBM (`{2,1,0}` = last dim contiguous,
  `{0,2,1}` = first dim contiguous). The GMM kernel reads its weight tiles expecting
  `{2,1,0}`, but the experts are **stored** as `{0,2,1}`, so XLA inserts a **layout-conversion
  copy of every expert weight** (wi_0/wi_1/wo × 36 layers ≈ 108 copies, ~530 MB each).
- Because the weights enter the compiled forward as **inputs** with their stored layout fixed,
  XLA re-does the reshuffle **inside every forward step** — the same weight is relaid-out 65×
  in a 65-step run, identical result each time. Pure, removable waste.
- **Fix:** store the experts on-device already in `{2,1,0}` (a layout hint / pre-arranged array
  at load). Then the ~108 relayout copies/step vanish → the copy category collapses toward the
  Qwen3-MoE residual (single digits). Load-time change, low risk, biggest single win.

## Bottleneck #2 — the 4× KV cache (long-context / capacity)

Per token per layer (K+V, 8 KV heads):

| | layout | bytes |
|---|---|---|
| **SGLang-JAX** | 8 heads × 2(K,V) × **128 (padded)** × 2 (bf16) | **4096 B** |
| **vLLM hd64** | 8 heads × **128 (K+V *packed*)** × 1 (fp8) | **1024 B** |

→ **4× = 2× (head_dim pad) × 2× (bf16 vs fp8).** Consequences: SGLang-JAX fits ~4× fewer
concurrent tokens (it literally could not run the in8192/out1024 **c=256** point — ~325 GB of
KV won't fit; vLLM's ~82 GB does) and reads 4× the KV bandwidth at decode.

### Why the kernel pads head_dim to 128 (the design rationale)

- **Hardware:** the TPU vector unit uses **(8 sublane × 128 lane)** tiles and a **128×128 MXU**;
  a matmul is full-utilization only when the contraction/output dim is a multiple of 128.
- **Attention:** head_dim is *both* the QKᵀ contraction dim and the ·V output dim, so the
  natural layout maps **head_dim → the 128-lane dim**. For head_dim=128 (Llama/Qwen) that's a
  perfect fit and one simple generic kernel handles everything.
- **head_dim=64 then has two options:** **pad 64→128** with zeros (simple, generic, 50% waste —
  SGLang-JAX's RPA v3 does `align_to(head_dim,128)`), or **pack** two 64-wide things (K+V, or
  two heads) into the 128 lanes (full utilization, but a *dedicated* kernel — vLLM's `hd64`).
- The KV **cache** is stored padded-to-128 as a *consequence* — the cache row mirrors the MXU
  tile so it DMAs into VMEM with no relayout on read. (Contrast bottleneck #1, where the stored
  layout *didn't* match the kernel → per-step relayout.)

### Can SGLang-JAX fix the 4× "completely"? Yes — but it's kernel work

- **2× from padding:** adopt vLLM's **packing** layout (fill the 128 lanes with K+V / two heads
  instead of zeros) — i.e. port a `ragged_paged_attention_hd64` + matching cache layout.
  Removing the `%128==0` assert alone does nothing (the kernel would still `align_to` 128).
- **2× from dtype:** add **fp8 KV** read/write to that kernel (SGLang-JAX exposes
  `--kv-cache-dtype` but fp8 isn't wired for the fused-KV RPA layout).

Both are medium-high Pallas effort, but **proven feasible — vLLM shipped exactly this.**

## Why Qwen3-MoE wins but gpt-oss loses (the control experiment)

Qwen3-30B-A3B, *same infra, same TP=8, same EPMoE path, same profiling method*:

| Category | **Qwen3-MoE-30B** | **gpt-oss-120b** |
|---|---|---|
| matmul/fusion | 40.3% | 30.6% |
| moe/gmm | 23.8% | 10.5% |
| collective (TP=8) | 14.9% | 16.2% |
| **memory/layout (copies)** | **9.4%** | **34.6%** |
| attention | 7.7% | 5.5% |

Qwen3-MoE spends ~64% on useful compute and only **9.4% on copies** → **SGLang-JAX's MoE path
is fine.** Both models do the *same* per-step expert relayout, but gpt-oss's experts are ~10×
larger bytes (120B vs 30B; `[128,2880,360]` @530 MB vs `[128,2048,96]` @50 MB), so it dominates
gpt-oss (34.6%) and is negligible for Qwen3-MoE (9.4%). Qwen3-8B (dense) has *no* MoE at all
and `head_dim=128` natively — which is why SGLang-JAX beat vLLM there but loses on gpt-oss.

## Head-stitch experiment (head_dim 64→128) — a negative result

A distilled "head-stitched" checkpoint (head_dim 64→**128**, heads 64→32 / KV 8→4) makes
head_dim native-128 (no padding). Measured with dummy weights (latency is weight-value-
independent), same workload:

| | head_dim=64 (stock) | head_dim=128 (stitched) |
|---|---|---|
| decode tok/s (bs64) | **1183** | **1175** (−0.7%, noise) |
| copies | 36.1% | 36.3% |
| KV capacity (max tokens) | 3.63 M | 3.63 M |

**No improvement**, because: (a) the 36% relayout copies are head_dim-independent (same
experts); (b) at TP=8 the 4 KV heads **replicate back to 8** (4 < 8 devices), negating the KV
saving — it would only help at TP≤4; (c) attention FLOP savings are ~nil at short context.
Conclusion: head-stitch is not the lever at TP=8; the KV win needs **fp8 + native hd64**, not
fewer heads.

## Ranked fix roadmap

| # | Fix | Wins | Effort / risk | When it helps |
|---|---|---|---|---|
| 1 | **Store expert weights in `{2,1,0}`** (kill per-step relayout) | ~25–30 pts of decode time | low, load-time | everywhere |
| 2 | **DP/EP parallelism** (TP=2×DP=4×EP vs TP=8) | ~16% collectives + reshard | med, mostly config | everywhere |
| 3 | **fp8 KV + native hd64 kernel** (packing, no pad) | 4× KV → capacity + bandwidth | med-high Pallas | long context / high concurrency |
| 4 | **Native MXFP4 experts** (w4a16, keep 4-bit) | 4× expert HBM footprint/bandwidth | high | decode, HBM capacity |
| — | Fix `gmm_v2` bf16 (drop `force_gmm_v1`) | MoE kernel speed | med Pallas | MoE-bound regimes |

Bottleneck #1 shows up in *every* regime and is the cheapest; #3/#4 matter most where vLLM's
lead is largest (long context, high concurrency).

## Appendix — reproducing the profiles

Capture with `bench_one_batch --profile` (bs=64, in=1024, out=64, bf16, tp=8) per model, then
aggregate the xprof `trace.json.gz` device (`/device:TPU:* / XLA Ops`) rows by op name and
bucket into categories (copy / matmul-fusion / collective / gmm / attention). The head-stitch
runs use `--load-format dummy` + `--json-model-override-args '{"head_dim":128,
"num_attention_heads":32,"num_key_value_heads":4}'` (the loader's dummy branch — no weights
needed). Raw traces are large (~36 MB each) and not committed; all numbers above are inline.
