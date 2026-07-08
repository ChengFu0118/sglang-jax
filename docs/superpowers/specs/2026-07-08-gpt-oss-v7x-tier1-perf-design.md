# Design: gpt-oss-120b Tier-1 performance on TPU v7x (parallelism + reshard/copy reduction)

Status: **proposed** (awaiting review)
Date: 2026-07-08
Branch: `gpt-oss-support` (fork)

## Context

gpt-oss now runs correctly in bf16 on TPU v7x (bf16 NaN fixed via `force_gmm_v1`), and the
head-to-head vs vLLM `tpu-inference` is done (`gpt-oss-roadmap/gptoss_120b_benchmark.md`):
**vLLM leads ~3.6–6.7× on output throughput and ~3–5× on ITL.** This spec covers closing that
gap.

### Why the gap (profile evidence)

An xprof trace of gpt-oss-120b on sgl-jax (`bench_one_batch --profile`, bs=64, in=1024,
out=64, TP=8, bf16; summed device time over 8 cores, prefill + 64 decode steps):

| Category | Device time | % |
|---|---|---|
| **memory/layout — `copy.*` (4,144 distinct ops)** | 5,815 ms | **34.6%** |
| matmul/fusion | 5,128 ms | 30.6% |
| **collective — `psum.*` / `all-reduce` (TP=8)** | 2,723 ms | 16.2% |
| moe/gmm (Pallas) | 1,758 ms | 10.5% |
| attention (RPA v3, Pallas) | 916 ms | 5.5% |
| other | 444 ms | 2.6% |

**Diagnosis:** at serving decode batch sizes the useful matmuls are small/memory-bound, and the
run is dominated by **8-way tensor-parallel overhead** — reshard/layout **copies (~28–35%)** and
**collectives (~16%)** — not by the MoE or attention kernels (which are the usual suspects but
are minor here). This is exactly why vLLM's **TP=2 × DP=4** wins: TP=2 collectives are far
cheaper than TP=8, and the 4 data-parallel replicas never synchronize during the forward pass.

### Framework facts (verified)

- `--tp-size` = total JAX devices; `--dp-size` = size of the `data` mesh axis;
  `attention_tp_size = tp_size // dp_size`. So `--tp-size 8 --dp-size 4` builds a
  `data=4 × tensor=2` mesh — attention runs data-parallel, structurally like vLLM's TP=2×DP=4.
  The KV pool already shards on the `data` axis (`attention_data_partition_axis="data"`), and
  pool sizes are aligned to `page_size * dp_size`.
- gpt-oss's MoE (`EPMoE`) builds a **separate `moe_mesh`** = `reshape(ep_size, tensor)` over all
  world devices and runs the experts under it via `shard_map`, `reshard`-ing `hidden_states`
  into/out of that mesh **every layer** (`reshard(hidden_states, P(None))` in, `reshard(result,
  out_sharding)` out, under `use_abstract_mesh`). gpt-oss currently uses `ep_size=1` → experts
  replicated, sharded on `tensor` across all 8 devices. This per-layer model-mesh↔moe_mesh
  round-trip is the prime suspect for the 4,144 copies, and under DP it would all-gather MoE
  inputs across the `data` axis unless `ep_size` is aligned to `dp_size`.

## Goal / success criteria

- **Primary:** materially raise gpt-oss-120b decode throughput and lower ITL on v7x-8 by cutting
  the parallelism overhead. Target: in a re-profile of the best config, the
  **collective + copy share drops from ~44% to < 20%**.
- **Correctness:** output stays coherent ("Paris" / "4" / "the lazy dog"), no NaN.
- **No collateral damage:** changes are gpt-oss-scoped or opt-in; Qwen and other models are
  bit-for-bit unaffected; the 8 mxfp4 unit tests still pass.

Out of scope (Tier 2, revisit after re-profile): hd64 attention path, fused-MoE / gmm_v2 fix,
fp8 KV, native MXFP4 matmul.

## Approach

### Part A — Parallelism (config-first, code only where needed)

Experiment matrix on the 120b server at a fixed decode-heavy workload (random, in=1024,
out=256, `bench_serving` max-concurrency ∈ {32, 64, 128}), capturing decode tok/s + ITL and an
xprof re-profile of the best point:

| Config | mesh (data×tensor) | MoE | note |
|---|---|---|---|
| tp8 / dp1 / ep1 | 1×8 | replicated, tensor=8 | **baseline** (current) |
| tp8 / dp4 / ep1 | 4×2 | replicated on moe_mesh(1,8) | attention DP; MoE may all-gather across data |
| tp8 / dp4 / ep4 | 4×2 | expert-parallel, expert=data | **expected winner** — mirrors vLLM |
| tp8 / dp2 / ep2 | 2×4 | expert=data | intermediate |
| tp8 / dp8 / ep8 | 8×1 | pure EP | attention fully DP |

**Anticipated code:** align the EPMoE `moe_mesh` with the DP model mesh so the `expert` axis maps
onto the `data` devices (i.e. drive `ep_size` from `dp_size`), avoiding a full all-gather of MoE
inputs across `data`. Fix any gpt-oss code that assumes `dp_size == 1`. Keep gpt-oss-scoped.

### Part B — Reshard / copy reduction (code)

Re-profile the Part-A winner. Expected dominant remaining cost = the per-layer
model-mesh↔moe_mesh round-trip + residual/`make_reduce_sharding` reshards. Reduce by:
- Keeping the MoE on the model mesh (or minimizing the `reshard(..., P(None))` replicate) so the
  MoE doesn't switch meshes each layer; and
- Dropping redundant residual reshards where the sharding is already correct.

Validate each change with a coherence check + an xprof re-profile confirming the copy share drops.

## Testing

- `bench_serving` decode throughput + ITL, before vs after, at the fixed workload above.
- xprof re-profile (`analyze_xprof` / the Chrome-trace parser) — category %-shift.
- Coherence: `/v1/completions` → "Paris" / "4" / "the lazy dog"; no NaN.
- `python -m unittest sgl_jax.test.test_mxfp4_gpt_oss` (8 tests).

## Risks

- DP/EP mesh interactions may need real (not just config) fixes — the sweep tells us how much is
  free config vs code.
- Copy reduction touches shared `moe.py` — keep changes gpt-oss-scoped / opt-in to avoid Qwen
  regressions; re-verify a non-gpt-oss MoE model if `moe.py` shared paths change.
- v7x is untuned generally; some overhead is systemic and may cap the achievable gain (documented
  honestly, not hidden).
