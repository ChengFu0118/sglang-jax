# sglang-jax vs vLLM tpu-inference — Model Support & Benchmark

Comparison of model coverage between **[sglang-jax](https://github.com/sgl-project/sglang-jax)**
and **[vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)**, plus a
head-to-head throughput benchmark on TPU v7x.

> **Sizes:** MoE models are shown as **total / active** parameters. `~` = approximate /
> best-known (family listed but exact checkpoint sizes vary). Dense models show total params.

> **Coverage model differs between the two engines:**
> - **sglang-jax** = a *hard allowlist* — a model runs only if it has a registered `EntryClass`
>   in `python/sgl_jax/srt/models/`.
> - **tpu-inference** = "*any vLLM model out-of-the-box*", but only the matrix below is
>   *validated*. Real coverage is broader than what's listed; sglang-jax's is exactly the list.

---

## ✅ Supported by both — identical checkpoints (safe for apples-to-apples)

| Model | Size(s) | sglang-jax | tpu-inference |
|---|---|---|---|
| **Qwen3 (dense)** | 0.6B / 1.7B / 4B / 8B / 14B / 32B | ✅ | ✅ |
| **Qwen3-MoE** | 30B-A3B (30B / 3B), 235B-A22B (235B / 22B) | ✅ | ✅ |
| **Qwen3.5** | ~9B dense · 397B-A17B (397B / 17B) MoE | ✅ | ✅ |
| **Qwen2.5-VL** | 3B / 7B / 32B / 72B | ✅ | ✅ |
| **Llama 3.x** | 3.1: 8B / 70B / 405B · 3.3: 70B | ✅ | ✅ |

**Best benchmark target: Qwen3** — both engines expose the exact same checkpoints
(Qwen3-8B, Qwen3-32B dense; Qwen3-30B-A3B MoE).

## ⚠️ Supported by both — same family, different checkpoint/version

| Family | sglang-jax has | tpu-inference has | Size(s) |
|---|---|---|---|
| **Gemma** | Gemma 2, **Gemma 4** | Gemma 3, **Gemma 4** | G2: 2B/9B/27B · G3: 1B/4B/12B/27B · G4: E2B / E4B / 26B-A4B / 31B |
| **DeepSeek V3** | V2, V3 | R1, V3.1 / V3.2 | V3/R1: 671B-A37B (671B / 37B) · V2: 236B / 21B · V2-Lite: 16B / 2.4B |
| **GLM** | GLM-4 MoE, **GLM-5** | **GLM-5** | GLM-4.5: 355B-A32B · GLM-4.5-Air: 106B-A12B · GLM-5: ~large MoE |
| **MiniMax** | M2 | M2.5 | ~230B-A10B (230B / 10B) |
| **Kimi** | Kimi-Linear, K2.5-VL | K2.6, K2-Thinking | Kimi-Linear: 48B-A3B · Kimi-K2: 1T-A32B (1T / 32B) |

## 🟦 sglang-jax only

| Model | Size(s) |
|---|---|
| Grok-1 / Grok-2 | Grok-1: 314B MoE (~86B active) · Grok-2: larger |
| Bailing MoE / V2 / V2.5 (Ling) | Ling-lite: 16.8B-A2.75B · Ling-plus: ~290B-A28.8B |
| MiMo-7B / V2-Flash / V2-Pro | MiMo-7B: 7B · V2-Flash / V2-Pro: ~MoE (Xiaomi) |
| Qwen1 / Qwen2 (+ Qwen2-MoE) | Qwen2: 0.5B–72B · Qwen2-MoE (A14B): 57B-A14B |
| Phi-3 | mini 3.8B / small 7B / medium 14B |
| InternLM3 | 8B |
| UMT5 | encoder-decoder (base→XXL) |
| Wan 2.1 / 2.2 (text-to-video) | 1.3B / 14B |

## 🟥 tpu-inference only

| Model | Size(s) |
|---|---|
| **gpt-oss** | 20b: 21B-A3.6B (21B / 3.6B) · 120b: 117B-A5.1B (117B / 5.1B) |
| Qwen3-Coder | 480B-A35B (480B / 35B) |
| Qwen3-Embedding | 8B |
| Qwen3-Omni / Qwen3-VL | Omni: 30B-A3B · VL: ~8B |
| DeepSeek-OCR | ~3B |
| Gemma 3 | 1B / 4B / 12B / 27B |

*(plus anything vLLM supports generically but hasn't validated on TPU)*

---

## Benchmark result (TPU v7x-8 / Ironwood, Qwen3-8B, tp=8, bf16)

Single v7x-8 host (4 chips × 2 = 8 JAX devices), prefix-cache off, identical client
(`sgl_jax.bench_serving`). Workload: random ISL 1024 / OSL 128.
- **sglang-jax** = repo `main` (built from source)
- **tpu-inference** = `vllm/vllm-tpu:nightly`

### Output token throughput (tok/s) — higher is better
| concurrency | sglang-jax | tpu-inference | sglang / vLLM |
|---:|---:|---:|:--|
| 16 | **1147.9** | 876.8 | **1.31×** |
| 32 | **2080.0** | 1684.5 | **1.23×** |
| 64 | **2934.4** | 2544.5 | **1.15×** |
| 128 | **4321.2** | 3949.4 | **1.09×** |
| 256 | 3492.8 | **4399.8** | 0.79× |

**Takeaway:** sglang-jax leads decode throughput & inter-token latency at low-to-mid
concurrency (16–128, by 9–31%); tpu-inference overtakes it at max concurrency (256).
The high-concurrency regression coincides with `RPA v3 tuned-block-size LOOKUP MISS
device=TPU v7` warnings — sglang-jax's attention kernels don't yet ship v7x-tuned block
sizes and fall back to a generic heuristic. On v6e (the repo's published data) sglang-jax
leads at all concurrencies.

Raw per-point data: `sweep_sglang.txt`, `sweep_vllm.txt`, `COMPARISON.md`.

## Notes

- **gpt-oss-120b cannot be compared** — tpu-inference only; sglang-jax has no gpt-oss model
  module (only a comment referencing its activation in a MoE kernel).
- Sizes for the newest checkpoints (GLM-5, Qwen3.5, MiniMax-M2.5, Kimi-K2.6, MiMo-V2,
  Bailing-2.6, Grok-2) are approximate — verify against the model card before relying on them.
