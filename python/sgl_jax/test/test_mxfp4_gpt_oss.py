"""Unit tests for the gpt-oss building blocks: MXFP4 dequant + swigluoai.

Run (from python/sgl_jax/test):
    python -m unittest test_mxfp4_gpt_oss
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.moe import _swigluoai
from sgl_jax.srt.utils.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    dequantize_mxfp4,
    e8m0_to_fp32,
    u8_unpack_e2m1,
)

# The 8 magnitudes representable by e2m1 (4-bit float, 2 exp / 1 mantissa bits).
# All are exact in bf16, and so are these times a power of two.
_E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _pack_e2m1_block(values_f32: jax.Array) -> jax.Array:
    """Quantize an fp32 vector (multiple of 32) to e2m1 and pack two per uint8."""
    codes = values_f32.astype(jnp.float4_e2m1fn)
    # bitcast two 4-bit codes into one uint8 (last dim halves).
    packed = jax.lax.bitcast_convert_type(codes.reshape(codes.shape[:-1] + (-1, 2)), jnp.uint8)
    return packed


def _e2m1_supported() -> bool:
    """Some backends (e.g. plain CPU) can't bitcast to/from float4_e2m1fn."""
    try:
        _pack_e2m1_block(jnp.zeros((32,), dtype=jnp.float32)).block_until_ready()
        return True
    except Exception:
        return False


_SKIP_E2M1 = None if _e2m1_supported() else "float4_e2m1fn bitcast unsupported on this backend"


class MXFP4DtypeTest(unittest.TestCase):
    def test_dtypes_exist(self):
        # These are required by the dequant path; fail loudly if the JAX build
        # doesn't expose them.
        self.assertTrue(hasattr(jnp, "float4_e2m1fn"))
        self.assertTrue(hasattr(jnp, "float8_e8m0fnu"))
        self.assertEqual(MXFP4_BLOCK_SIZE, 32)

    def test_e8m0_matches_ldexp(self):
        bias = -int(jnp.finfo(jnp.float8_e8m0fnu).minexp)
        us = jnp.array([bias - 3, bias, bias + 1, bias + 5], dtype=jnp.uint8)
        expected = np.array([2.0 ** (-3), 1.0, 2.0, 2.0**5], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(e8m0_to_fp32(us)), expected)


@unittest.skipIf(_SKIP_E2M1 is not None, _SKIP_E2M1 or "")
class MXFP4DequantTest(unittest.TestCase):
    def test_unpack_roundtrip_identity(self):
        # Two blocks of 32 e2m1-representable values (mix of signs).
        base = np.array(
            _E2M1_MAGNITUDES + [-v for v in _E2M1_MAGNITUDES],
            dtype=np.float32,
        )  # length 16
        vals = np.concatenate([base, base, base, base])  # length 64 == 2 blocks
        vals_j = jnp.asarray(vals)

        codes = u8_unpack_e2m1(_pack_e2m1_block(vals_j)).astype(jnp.float32)
        np.testing.assert_array_equal(np.asarray(codes), vals)

    def test_dequant_unit_scale(self):
        bias = -int(jnp.finfo(jnp.float8_e8m0fnu).minexp)  # u == bias -> scale 2**0 == 1
        base = np.array(_E2M1_MAGNITUDES + [-v for v in _E2M1_MAGNITUDES], dtype=np.float32)
        vals = np.concatenate([base, base])  # 32 values == one block
        blocks_u8 = _pack_e2m1_block(jnp.asarray(vals))[None, :]  # [1, 16]
        scales_u8 = jnp.full((1,), bias, dtype=jnp.uint8)

        out = dequantize_mxfp4(blocks_u8, scales_u8)
        self.assertEqual(out.dtype, jnp.bfloat16)
        np.testing.assert_allclose(np.asarray(out).astype(np.float32), vals, atol=0.0)

    def test_dequant_power_of_two_scale(self):
        bias = -int(jnp.finfo(jnp.float8_e8m0fnu).minexp)
        exponent = 2
        scale = float(2**exponent)
        base = np.array(_E2M1_MAGNITUDES + [-v for v in _E2M1_MAGNITUDES], dtype=np.float32)
        vals = np.concatenate([base, base])  # one block of 32
        blocks_u8 = _pack_e2m1_block(jnp.asarray(vals))[None, :]
        scales_u8 = jnp.full((1,), bias + exponent, dtype=jnp.uint8)

        out = np.asarray(dequantize_mxfp4(blocks_u8, scales_u8)).astype(np.float32)
        np.testing.assert_allclose(out, vals * scale, atol=0.0)


class InterleaveSplitTest(unittest.TestCase):
    def test_gate_up_interleave_split(self):
        # Sanity check the even/odd split convention the loader relies on:
        # gate = rows[::2], up = rows[1::2] along the fused output axis.
        e, two_inter, hidden = 2, 8, 3
        gate_up = np.arange(e * two_inter * hidden, dtype=np.float32).reshape(e, two_inter, hidden)
        gate = gate_up[:, 0::2, :]
        up = gate_up[:, 1::2, :]
        self.assertEqual(gate.shape, (e, two_inter // 2, hidden))
        np.testing.assert_array_equal(gate[0, 0], gate_up[0, 0])
        np.testing.assert_array_equal(up[0, 0], gate_up[0, 1])


class SwigluOAITest(unittest.TestCase):
    def test_matches_kernel_reference(self):
        try:
            from sgl_jax.srt.kernels.fused_moe.v2.kernel import swigluoai as kernel_swigluoai
        except Exception as e:  # pragma: no cover - kernel import needs pallas
            self.skipTest(f"could not import v2 kernel swigluoai: {e}")

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        gate = jax.random.normal(k1, (16, 32)) * 10  # large values exercise the clamp
        up = jax.random.normal(k2, (16, 32)) * 10

        ours = _swigluoai(gate, up, alpha=1.702, limit=7.0)
        ref = kernel_swigluoai(gate, up, alpha=1.702, limit=7.0)
        np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), rtol=1e-6, atol=1e-6)

    def test_clamp_and_plus_one(self):
        # gate above limit is clamped; up is clamped to [-limit, limit]; the
        # linear branch is (up + 1).
        limit = 7.0
        alpha = 1.702
        gate = jnp.array([[100.0]])
        up = jnp.array([[100.0]])
        out = np.asarray(_swigluoai(gate, up, alpha=alpha, limit=limit))
        glu = limit * (1.0 / (1.0 + np.exp(-alpha * limit)))
        expected = (limit + 1.0) * glu
        np.testing.assert_allclose(out, [[expected]], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
