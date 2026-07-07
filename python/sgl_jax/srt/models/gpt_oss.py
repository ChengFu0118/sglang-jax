"""GPT-OSS (openai/gpt-oss-20b / -120b) model for SGLang-JAX.

gpt-oss is a Mixture-of-Experts decoder with a few distinctive pieces, all of
which reuse existing SGLang-JAX primitives:

* **Attention sinks** — a per-head learned logit added as a phantom token in the
  softmax denominator (RPA v3 / FlashAttention backend, ``attention_sink=``).
* **Alternating sliding/full attention** — even layers use a 128-token sliding
  window, odd layers are full (``config.layer_types``).
* **QKV/O bias** and **YaRN RoPE** (``get_rope("yarn")``).
* **MoE** with a router bias, top-k *then* softmax, per-expert projection
  biases, and a clamped ``swigluoai`` activation with ``(up + 1)`` — handled by
  the extended :class:`EPMoE` (``activation="swigluoai"``, ``use_expert_bias``).

Weights ship as MXFP4 (4-bit e2m1 codes + e8m0 group-32 scales) for the experts
only; everything else is bf16. We dequantize the experts to bf16 at load time
(:func:`dequantize_mxfp4`) — native FP4 matmul is a follow-up.
"""

import glob
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.parallel_utils import make_reduce_sharding
from sgl_jax.srt.utils.quantization.mxfp4 import dequantize_mxfp4

logger = logging.getLogger(__name__)



def _is_sliding_layer(config: PretrainedConfig, layer_id: int) -> bool:
    """Whether ``layer_id`` uses sliding-window attention.

    gpt-oss lists this per layer in ``config.layer_types``; fall back to the
    even-layer convention (``i % 2 == 0``) used by the reference implementation.
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and layer_id < len(layer_types):
        return layer_types[layer_id] == "sliding_attention"
    return layer_id % 2 == 0


class GptOssAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh

        hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        self.q_head_num = config.num_attention_heads
        self.kv_head_num = config.num_key_value_heads
        self.q_size = self.q_head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.scaling = self.head_dim**-0.5

        attention_bias = getattr(config, "attention_bias", True)

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=self.q_head_num * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 131072),
            base=getattr(config, "rope_theta", 150000),
            is_neox_style=True,
            rope_scaling=getattr(config, "rope_scaling", None),
            dtype=dtype,
        )

        # The RPA v3 kernel's internal head_dim<128 padding path appears to
        # mishandle gpt-oss's head_dim=64 (produces NaN). Pad q/k/v to 128
        # ourselves (with zeros — identical KV content) so the kernel runs its
        # well-tested head_dim=128 path. Softmax scale stays 1/sqrt(real 64).
        self.kernel_head_dim = ((self.head_dim + 127) // 128) * 128
        sliding = _is_sliding_layer(config, layer_id)
        # gpt-oss has massive-activation outliers; run the attention softmax in
        # fp32 so bf16 serving doesn't blow up to NaN.
        self.attn = RadixAttention(
            num_heads=self.q_head_num,
            head_dim=self.kernel_head_dim,
            scaling=self.scaling,
            num_kv_heads=self.kv_head_num,
            layer_id=layer_id,
            sliding_window_size=(getattr(config, "sliding_window", 0) if sliding else 0),
            softmax_dtype=jnp.float32,
        )

        # Per-head attention sink logit. Stored in f32 (the RPA v3 kernel upcasts
        # anyway); sharded across heads on the "tensor" axis.
        self.sinks = nnx.Param(
            jnp.zeros((self.q_head_num,), dtype=jnp.float32, out_sharding=P("tensor"))
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor")),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor")),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor")),
        )

        q, k = self.rotary_emb(positions, q, k)

        # Pad head_dim 64 -> 128 (zeros) to use the kernel's head_dim=128 path.
        if self.kernel_head_dim != self.head_dim:
            pad = ((0, 0), (0, 0), (0, self.kernel_head_dim - self.head_dim))
            q = jnp.pad(q, pad)
            k = jnp.pad(k, pad)
            v = jnp.pad(v, pad)

        attn_output, kv_fused = self.attn(
            q,
            k,
            v,
            forward_batch,
            token_to_kv_pool,
            attention_sink=self.sinks.value,
        )

        # attn_output is [tokens, q_head_num * kernel_head_dim]; slice each head
        # back to the real head_dim before o_proj.
        if self.kernel_head_dim != self.head_dim:
            attn_output = attn_output.reshape(-1, self.q_head_num, self.kernel_head_dim)[
                ..., : self.head_dim
            ].reshape(-1, self.q_head_num * self.head_dim)
        output, _ = self.o_proj(attn_output, out_sharding=out_sharding)
        return output, kv_fused


class GptOssMoE(nnx.Module):
    """gpt-oss MoE block: router (bias + top-k + post-top-k softmax) → EPMoE."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = getattr(
            config, "num_experts_per_tok", getattr(config, "experts_per_token", 4)
        )

        # score_func=None: gpt-oss softmaxes *after* top-k (done in __call__),
        # not over the full logit vector. enable_expert_bias holds the additive
        # router bias (added to logits pre-top-k).
        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=self.num_experts,
            weight_dtype=dtype,
            enable_expert_bias=True,
            score_func=None,
        )

        self.experts = EPMoE(
            hidden_size=config.hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            intermediate_dim=config.intermediate_size,
            mesh=mesh,
            activation="swigluoai",
            ep_size=getattr(config, "ep_size", 1),
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            quantization_config=None,  # MXFP4 is dequantized to bf16 at load.
            use_expert_bias=True,
            swiglu_limit=getattr(config, "swiglu_limit", 7.0),
            swiglu_alpha=1.702,
            # The megablox v2 grouped-matmul kernel emits NaNs in bf16 when a
            # padded (short) prompt routes all its padding tokens to one expert
            # (extremely imbalanced group_sizes). This is *the* gpt-oss bf16
            # serving NaN. v1 is numerically correct for the same inputs.
            force_gmm_v1=True,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        *,
        out_sharding: jax.sharding.NamedSharding,
    ) -> tuple[jax.Array, jax.Array]:
        # Zero padded (invalid) tokens before the MoE. gpt-oss's padded-token
        # activations can be NaN, and EPMoE's grouped matmul mixes all tokens,
        # so an unmasked padded row corrupts real tokens' outputs (only shows up
        # with padding — a short prompt padded to a bucket). The fused-MoE path
        # guards this via token_valid_mask; replicate it here.
        token_valid_mask = forward_batch.get_token_valid_mask(
            hidden_states.shape[0],
            out_sharding=NamedSharding(self.mesh, P(out_sharding.spec[0])),
        )
        if token_valid_mask is not None:
            hidden_states = jnp.where(token_valid_mask[:, None], hidden_states, 0.0)

        router_logits = self.moe_gate(hidden_states)  # raw logits, f32
        router_logits = router_logits + self.moe_gate.bias.value.astype(router_logits.dtype)

        topk_weights, topk_ids = jax.lax.top_k(router_logits, self.num_experts_per_tok)
        # gpt-oss normalizes with softmax over the selected experts only.
        topk_weights = jax.nn.softmax(topk_weights.astype(jnp.float32), axis=-1)

        mlp_output = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            out_sharding=out_sharding,
        )
        return mlp_output, topk_ids


class GptOssDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.enable_sequence_parallel = getattr(config, "enable_sequence_parallel", False)

        self.self_attn = GptOssAttention(config, layer_id=layer_id, mesh=mesh, dtype=dtype)
        self.mlp = GptOssMoE(config, layer_id=layer_id, mesh=mesh, dtype=dtype)

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        reduce_sharding = make_reduce_sharding(
            hidden_states, self.mesh, enable_sp=self.enable_sequence_parallel
        )

        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            out_sharding=reduce_sharding,
        )
        hidden_states += jax.sharding.reshard(residual, reduce_sharding)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, topk_ids = self.mlp(
            hidden_states,
            forward_batch,
            out_sharding=reduce_sharding,
        )
        residual = jax.sharding.reshard(residual, reduce_sharding)

        return hidden_states, residual, kv_fused, topk_ids


class GptOssModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                GptOssDecoderLayer(config=config, layer_id=i, dtype=dtype, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[jax.Array], list[jax.Array]]:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_topk_ids


class GptOssForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = GptOssModel(config, dtype=self.dtype, mesh=mesh)

        # gpt-oss has tie_word_embeddings=False; a separate lm_head is expected.
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    # ------------------------------------------------------------------
    # Weight loading (custom): MXFP4 experts + fused/interleaved gate_up.
    # ------------------------------------------------------------------

    def load_weights(self, model_config: ModelConfig):
        self._st_handles: dict[str, safe_open] = {}
        key_to_file: dict[str, str] = {}
        model_path = model_config.model_path
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise RuntimeError(f"No *.safetensors files found in {model_path}")
        for f in files:
            handle = safe_open(f, framework="np", device="cpu")
            self._st_handles[f] = handle
            for k in handle.keys():
                key_to_file[k] = f

        import ml_dtypes

        def get(name: str) -> np.ndarray:
            # safetensors' numpy framework returns bf16 tensors as a raw 2-byte
            # void dtype ("V2"); reinterpret them as bfloat16 (uint8 MXFP4 blocks
            # and scales come through unchanged).
            arr = np.asarray(self._st_handles[key_to_file[name]].get_tensor(name))
            if arr.dtype == np.dtype("V2"):
                arr = arr.view(ml_dtypes.bfloat16)
            return arr

        num_layers = self.config.num_hidden_layers

        # Run all host-side dequant / transpose on CPU to avoid TPU OOM; only
        # the final sharded arrays land on device via make_array_from_callback.
        with jax.default_device(jax.devices("cpu")[0]):
            # --- Embeddings / final norm / LM head ---
            self._assign(
                self.model.embed_tokens.embedding,
                get("model.embed_tokens.weight"),
                ("tensor", None),
            )
            self._assign(self.model.norm.scale, get("model.norm.weight"), (None,))
            if not getattr(self.config, "tie_word_embeddings", False):
                self._assign(self.lm_head.embedding, get("lm_head.weight"), ("tensor", None))

            for i in range(num_layers):
                self._load_layer(i, get)

        for handle in self._st_handles.values():
            # safe_open closes on GC; drop refs so we don't hold file descriptors.
            del handle
        self._st_handles.clear()
        logger.info("GPT-OSS weights loaded successfully!")

    def _assign(
        self,
        param: nnx.Variable,
        host_array,
        spec,
        *,
        mesh: jax.sharding.Mesh | None = None,
        transpose: bool = False,
    ):
        """Assign a host array into an nnx param with an explicit sharding.

        ``eval_shape`` leaves the param with an *abstract*-mesh sharding that
        ``make_array_from_callback`` can't use, so we build a concrete
        ``NamedSharding`` from ``spec`` on ``mesh`` (the model mesh by default,
        or an EPMoE ``moe_mesh`` for expert tensors).
        """
        import ml_dtypes

        arr = np.asarray(host_array)
        if transpose:
            arr = arr.T
        target = param.value  # abstract ShapeDtypeStruct from eval_shape
        if tuple(arr.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch assigning weight: got {arr.shape}, expected {target.shape}"
            )
        target_dtype = jnp.dtype(target.dtype)
        np_dtype = ml_dtypes.bfloat16 if target_dtype == jnp.bfloat16 else np.dtype(target_dtype)
        arr = arr.astype(np_dtype)
        sharding = NamedSharding(mesh or self.mesh, P(*spec))
        param.value = jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])

    def _load_layer(self, i: int, get):
        prefix = f"model.layers.{i}"
        layer = self.model.layers[i]
        attn = layer.self_attn

        # Norms
        self._assign(layer.input_layernorm.scale, get(f"{prefix}.input_layernorm.weight"), (None,))
        self._assign(
            layer.post_attention_layernorm.scale,
            get(f"{prefix}.post_attention_layernorm.weight"),
            (None,),
        )

        # Attention projections (HF stores [out, in]; LinearBase wants [in, out]).
        col = (None, "tensor")  # column-parallel weight sharding
        row = ("tensor", None)  # row-parallel weight sharding
        sa = f"{prefix}.self_attn"
        self._assign(attn.q_proj.weight, get(f"{sa}.q_proj.weight"), col, transpose=True)
        self._assign(attn.k_proj.weight, get(f"{sa}.k_proj.weight"), col, transpose=True)
        self._assign(attn.v_proj.weight, get(f"{sa}.v_proj.weight"), col, transpose=True)
        self._assign(attn.o_proj.weight, get(f"{sa}.o_proj.weight"), row, transpose=True)
        self._assign(attn.q_proj.bias, get(f"{sa}.q_proj.bias"), ("tensor",))
        self._assign(attn.k_proj.bias, get(f"{sa}.k_proj.bias"), ("tensor",))
        self._assign(attn.v_proj.bias, get(f"{sa}.v_proj.bias"), ("tensor",))
        self._assign(attn.o_proj.bias, get(f"{sa}.o_proj.bias"), (None,))

        # Attention sinks (per-head, sharded on "tensor"; upcast to f32).
        self._assign(attn.sinks, get(f"{sa}.sinks").astype(np.float32), ("tensor",))

        # Router (GateLogit kernel is [hidden, experts] → transpose HF [experts, hidden]).
        self._assign(
            layer.mlp.moe_gate.kernel,
            get(f"{prefix}.mlp.router.weight"),
            (None, None),
            transpose=True,
        )
        self._assign(layer.mlp.moe_gate.bias, get(f"{prefix}.mlp.router.bias"), (None,))

        # Experts: MXFP4 dequant + interleaved gate/up split + transpose to
        # EPMoE [E, k, n] layout.
        self._load_experts(prefix, layer.mlp.experts, get)

    def _load_experts(self, prefix: str, experts: EPMoE, get):
        eprefix = f"{prefix}.mlp.experts"

        # gate_up_proj: MXFP4 [E, 2*inter, hidden] (out, in). Dequant then split
        # gate=even / up=odd rows along the (2*inter) output axis.
        gate_up = dequantize_mxfp4(
            jnp.asarray(get(f"{eprefix}.gate_up_proj_blocks")),
            jnp.asarray(get(f"{eprefix}.gate_up_proj_scales")),
        )  # [E, 2*inter, hidden] bf16
        # EPMoE wi_0/wi_1 are [E, hidden(k), inter(n)] → transpose (out, in)->(in, out).
        wi_0 = np.asarray(jnp.transpose(gate_up[:, 0::2, :], (0, 2, 1)))
        wi_1 = np.asarray(jnp.transpose(gate_up[:, 1::2, :], (0, 2, 1)))
        del gate_up

        # down_proj: MXFP4 [E, hidden(out), inter(in)]. EPMoE wo is [E, inter(k), hidden(n)].
        down = dequantize_mxfp4(
            jnp.asarray(get(f"{eprefix}.down_proj_blocks")),
            jnp.asarray(get(f"{eprefix}.down_proj_scales")),
        )
        wo = np.asarray(jnp.transpose(down, (0, 2, 1)))
        del down

        mm = experts.moe_mesh
        self._assign(experts.wi_0, wi_0, ("expert", None, "tensor"), mesh=mm)
        self._assign(experts.wi_1, wi_1, ("expert", None, "tensor"), mesh=mm)
        self._assign(experts.wo, wo, ("expert", "tensor", None), mesh=mm)

        # Biases: gate_up_proj_bias [E, 2*inter] split even/odd → [E, 1, inter];
        # down_proj_bias [E, hidden] → [E, 1, hidden]. Matches gmm rhs_bias
        # [num_groups, 1, out_dim] contract.
        gate_up_bias = np.asarray(get(f"{eprefix}.gate_up_proj_bias"))
        down_bias = np.asarray(get(f"{eprefix}.down_proj_bias"))
        gate_b = gate_up_bias[:, 0::2][:, None, :]
        up_b = gate_up_bias[:, 1::2][:, None, :]
        wo_b = down_bias[:, None, :]
        wi_bias_spec = ("expert", None, "tensor")
        self._assign(experts.w0_kernel_bias, gate_b, wi_bias_spec, mesh=mm)
        self._assign(experts.w1_kernel_bias, up_b, wi_bias_spec, mesh=mm)
        self._assign(experts.wo_kernel_bias, wo_b, ("expert", None, None), mesh=mm)

    # ------------------------------------------------------------------

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(forward_batch, kv_pool)

        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, {"token_to_kv_pool": layers_kv_fused}, True, layers_topk_ids

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.embedding.value
        if not getattr(self.config, "tie_word_embeddings", False):
            head = self.lm_head.embedding.value
        else:
            head = embed
        return embed, head


EntryClass = GptOssForCausalLM
