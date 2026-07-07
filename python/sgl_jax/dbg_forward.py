"""Minimal offline compiled-forward NaN probe for gpt-oss bf16 debugging.

Reuses bench_one_batch's load_model / synthetic inputs / extend / decode, but runs
a single prefill + a few decode steps and reports finite/NaN/amax at each stage.
Runs the exact compiled ModelRunner.forward path (RPA v3 kernel), single process,
so jax.debug.print / host reads work (unlike the server executor).

Usage:
  PYTHONPATH=/workspace/repo/python python3 -m sgl_jax.dbg_forward \
    --model-path /workspace/models/gpt-oss-20b --trust-remote-code \
    --device tpu --tp-size 8 --dtype bfloat16 \
    --json-model-override-args '{"num_hidden_layers":2}' \
    --attention-backend fa --context-length 1024
"""

import argparse
import logging

import jax
import numpy as np

from sgl_jax.srt.entrypoints.engine import _set_envs_and_config
from sgl_jax.srt.server_args import PortArgs, ServerArgs

import sgl_jax.bench_one_batch as b

# Extra knobs (parsed out before ServerArgs sees argv).
_EXTRA = {
    "--dbg-input-len": ("dbg_input_len", int, 5),
    "--dbg-decode-steps": ("dbg_decode_steps", int, 5),
    "--dbg-batch-size": ("dbg_batch_size", int, 1),
}


def _check(name, arr):
    a = np.array(jax.device_get(arr)).astype(np.float32)
    finite = bool(np.isfinite(a).all())
    nan = bool(np.isnan(a).any())
    amax = float(np.nanmax(np.abs(a))) if a.size else 0.0
    print(f"[NaNPROBE] {name:14s} shape={a.shape} finite={finite} nan={nan} amax={amax:.5g}",
          flush=True)
    return finite


def main():
    argv = __import__("sys").argv[1:]
    extra_vals = {}
    filtered = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _EXTRA:
            key, typ, _ = _EXTRA[tok]
            extra_vals[key] = typ(argv[i + 1])
            i += 2
            continue
        filtered.append(tok)
        i += 1
    for _flag, (key, _typ, default) in _EXTRA.items():
        extra_vals.setdefault(key, default)

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(filtered)
    server_args = ServerArgs.from_cli_args(args)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    model_runner, tokenizer = b.load_model(server_args, port_args, 0)

    reqs = b.prepare_synthetic_inputs_for_latency_test(
        extra_vals["dbg_batch_size"], extra_vals["dbg_input_len"]
    )
    print(f"[NaNPROBE] input_len={extra_vals['dbg_input_len']} "
          f"batch={extra_vals['dbg_batch_size']} "
          f"dtype={server_args.dtype} backend={server_args.attention_backend}", flush=True)

    next_token_ids, next_token_logits, batch = b.extend(reqs, model_runner)
    ok = _check("prefill", next_token_logits)
    nti_cpu = np.array(next_token_ids)
    for step in range(extra_vals["dbg_decode_steps"]):
        next_token_ids, next_token_logits = b.decode(nti_cpu, batch, model_runner)
        nti_cpu = np.array(next_token_ids)
        ok = _check(f"decode{step}", next_token_logits) and ok
    print(f"[NaNPROBE] RESULT all_finite={ok}", flush=True)


if __name__ == "__main__":
    jax.distributed.initialize()
    main()
