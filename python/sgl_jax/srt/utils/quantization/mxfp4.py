"""MXFP4 dequantization helpers.

gpt-oss ships its MoE expert weights as MXFP4: 4-bit ``e2m1`` codes packed two
per ``uint8`` byte, paired with an ``e8m0`` group-of-32 scale (also stored as
``uint8``). This module unpacks those tensors back to bf16 at load time so the
rest of the engine can treat the experts as ordinary bf16 weights.

Ported from tpu-inference ``layers/common/quantization/__init__.py``. Native
FP4 matmul (keeping the weights 4-bit on device) is an explicit follow-up.
"""

import jax
import jax.numpy as jnp

# MXFP4 quantizes in blocks of 32 elements along the contraction axis; each
# block shares a single e8m0 scale.
MXFP4_BLOCK_SIZE = 32


def u8_unpack_e2m1(u8_packed_e2m1: jax.Array) -> jax.Array:
    """Unpack an ``e2m1`` tensor that was packed two-per-byte into ``uint8``.

    ``bitcast_convert_type`` widens each ``uint8`` into two ``float4_e2m1fn``
    values (adding a trailing dim of size 2), which we then flatten back into
    the last axis — doubling its length.
    """
    assert u8_packed_e2m1.dtype == jnp.uint8, u8_packed_e2m1.dtype
    e2m1 = jax.lax.bitcast_convert_type(u8_packed_e2m1, jnp.float4_e2m1fn)
    return jnp.reshape(e2m1, e2m1.shape[:-2] + (-1,))


def e8m0_to_fp32(u8: jax.Array) -> jax.Array:
    """Convert an ``e8m0`` scale (bit-stored as ``uint8``) into fp32.

    ``e8m0`` is a pure power-of-two exponent with no mantissa, so the value is
    ``2 ** (stored - bias)``.
    """
    assert u8.dtype == jnp.uint8, u8.dtype
    e8_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    exponents = u8.astype(jnp.int32) + e8_finfo.minexp
    ones = jnp.ones_like(u8, dtype=jnp.float32)
    return jnp.ldexp(ones, exponents)


def dequantize_mxfp4(
    blocks_u8: jax.Array,
    scales_u8: jax.Array,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Dequantize a packed MXFP4 tensor to ``out_dtype``.

    Args:
        blocks_u8: ``uint8`` codes with shape ``[..., num_blocks, 16]`` — the
            last two axes describe the (block-quantized) contraction dim, where
            each of the 16 bytes holds two ``e2m1`` codes (32 codes per block).
        scales_u8: ``uint8`` ``e8m0`` scales with shape ``[..., num_blocks]``,
            one per 32-element block, broadcast across the block.
        out_dtype: Output dtype (bf16 by default).

    Returns:
        Dequantized tensor with shape ``[..., num_blocks * 32]`` (the packed
        contraction axis expanded back to full width).
    """
    # codes: [..., num_blocks, 16] -> [..., num_blocks, 32]
    codes = u8_unpack_e2m1(blocks_u8).astype(jnp.float32)
    # scales: [..., num_blocks] -> [..., num_blocks, 1] to broadcast over block.
    scales = e8m0_to_fp32(scales_u8)[..., None]

    dequant = codes * scales  # [..., num_blocks, 32]
    out_shape = dequant.shape[:-2] + (dequant.shape[-2] * dequant.shape[-1],)
    return dequant.reshape(out_shape).astype(out_dtype)
