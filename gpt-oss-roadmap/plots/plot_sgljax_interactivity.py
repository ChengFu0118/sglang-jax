#!/usr/bin/env python3
"""InferenceX-style throughput-per-accelerator vs interactivity plot for
gpt-oss-120b, featuring SGLang-JAX on TPU v7x (bf16), compared to NVIDIA
B200/GB200 and vLLM-TPU v7x (both FP4).

Axes (InferenceX): x = interactivity (output tok/s per user = 1000/TPOT_ms),
                   y = output throughput per accelerator (tok/s per GPU/chip).
Concurrency is the swept parameter tracing each curve. Per-chip normalization:
a v7x-8 node = 4 chips; SGLang-JAX runs TP=8 (one replica across all 4 chips) and
vLLM runs TP=2xDP=4 (4 one-chip replicas) -> both divide aggregate by 4 chips.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "..", "gptoss_120b_bench_data", "sglang", "gptoss120b_fix1_sweep.jsonl")
IX = json.load(open("/tmp/ix_gptoss_points.json"))
CHIPS = 4  # v7x-8 node = 4 chips (2 cores/chip)


def frontier(points):
    """Pareto upper envelope: max tput for interactivity >= x."""
    out, best = [], -1
    for iv, tp in sorted(points, key=lambda p: -p[0]):
        if tp > best:
            out.append((iv, tp)); best = tp
    return sorted(out)


# --- SGLang-JAX v7x (bf16), computed from the sweep JSONL (osl=1024 runs) ---
SGL = {}
for line in open(JSONL):
    if not line.strip():
        continue
    r = json.loads(line)
    if r["random_output_len"] != 1024:
        continue
    shp = f'{r["random_input_len"]}_1024'
    SGL.setdefault(shp, []).append(
        (1000.0 / r["median_tpot_ms"], r["output_throughput"] / CHIPS, r["max_concurrency"])
    )
for s in SGL:
    SGL[s].sort()

# --- vLLM-TPU v7x (FP4) reference points (per-chip), from the canonical skill script ---
VLLM = {
    "1024_1024": [(1000/20.82, 2992), (1000/20.13, 3070), (1000/12.52, 3365.53/4),
                  (1000/20.8, 10449.87/4), (1000/12.42, 4868.72/4)],
    "8192_1024": [(1000/37.18, 1656), (1000/35.97, 1699), (1000/13.88, 3888.83/4),
                  (1000/36.4, 6283.91/4), (1000/14.07, 3892.96/4)],
}

CURVES = {
    "b200-vllm":        ("NVIDIA B200 (vLLM, FP4)",         "#76b900", "o"),
    "b200-trt":         ("NVIDIA B200 (TRT-LLM, FP4)",      "#417505", "s"),
    "gb200-dynamo-trt": ("NVIDIA GB200 (Dynamo-TRT, FP4)",  "#1f77b4", "^"),
}
shapes = [("1024_1024", "1k in / 1k out"), ("8192_1024", "8k in / 1k out")]


def draw(ax, shp, logscale, zoom=False):
    # NVIDIA frontiers
    for key, (label, color, mk) in CURVES.items():
        if shp not in IX.get(key, {}):
            continue
        fr = frontier([[p[0], p[1]] for p in IX[key][shp]])
        if fr:
            xs, ys = zip(*fr)
            ax.plot(xs, ys, "-", color=color, lw=2, marker=mk, ms=4, label=label, alpha=0.9)
    # vLLM v7x FP4 reference (scatter)
    if shp in VLLM:
        vx = [p[0] for p in VLLM[shp]]; vy = [p[1] for p in VLLM[shp]]
        ax.scatter(vx, vy, s=70, marker="D", facecolors="none", edgecolors="#ff7f0e",
                   linewidths=1.4, label="vLLM-TPU v7x / chip (FP4)", zorder=5)
    # SGLang-JAX v7x bf16 (highlighted frontier)
    fr = frontier([(iv, tp) for iv, tp, _ in SGL[shp]])
    xs, ys = zip(*fr)
    ax.plot(xs, ys, "-", color="#d62728", lw=2.2, zorder=6)
    for iv, tp, c in SGL[shp]:
        ax.scatter([iv], [tp], s=150, marker="*", color="#d62728",
                   edgecolors="black", linewidths=0.7, zorder=7)
        ax.annotate(f"c{c}", (iv, tp), textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color="#333")
    ax.scatter([], [], s=150, marker="*", color="#d62728", edgecolors="black",
               label="SGLang-JAX v7x / chip (bf16, this work)")
    ax.set_title(f"gpt-oss-120b — {dict(shapes)[shp]}", fontsize=12, weight="bold")
    ax.set_xlabel("Interactivity  (output tok/s per user = 1000/TPOT)" + ("  [log]" if logscale else ""),
                  fontsize=10)
    ax.set_ylabel("Output throughput per accelerator (tok/s per GPU/chip)" + ("  [log]" if logscale else ""),
                  fontsize=10)
    if logscale:
        ax.set_xscale("log"); ax.set_yscale("log")
    if zoom:
        tpu_pts = [tp for _, tp, _ in SGL[shp]] + [p[1] for p in VLLM.get(shp, [])]
        ax.set_xlim(0, 130); ax.set_ylim(0, max(tpu_pts) * 1.15)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


for logscale, zoom, suffix in [(True, False, ""), (False, True, "_zoom")]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, (shp, _) in zip(axes, shapes):
        draw(ax, shp, logscale, zoom=zoom)
    extra = " — zoomed to TPU operating region (NVIDIA curves extend above)" if zoom else ""
    fig.suptitle("gpt-oss-120b serving: SGLang-JAX TPU v7x (bf16) vs NVIDIA B200/GB200 & vLLM-TPU (FP4)  "
                 "— InferenceX throughput-interactivity frontier (NVIDIA = SemiAnalysis InferenceX DB 2026-06-29)" + extra,
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, f"sgljax_v7x_interactivity{suffix}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)

# numeric summary
for shp, _ in shapes:
    print(f"\n== {shp} ==")
    best = max(SGL[shp], key=lambda p: p[1])
    print(f"  SGLang-JAX v7x bf16 best/chip: {best[1]:.0f} tok/s/chip @ intvty {best[0]:.1f} (conc {best[2]})")
    for key, (label, _, _) in CURVES.items():
        if shp in IX.get(key, {}):
            mx = max(IX[key][shp], key=lambda p: p[1])
            print(f"  {label}: {mx[1]:.0f} tok/s/gpu @ intvty {mx[0]:.0f} (conc {mx[2]})")
