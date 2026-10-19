"""Plot a sampler sweep: KID and FID against guidance, one line per step count,
and density against coverage with each point labelled by its guidance scale.
The real-vs-real floor is drawn in every panel.

Usage:
    python -m scripts.plot_sampler_metrics results/sampler_sweep_steps50.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.metrics import load_floor

SERIES = ("darkorange", "forestgreen")
INK = "#57606a"
FLOOR_COLOR = "#8a8a8a"
LABEL_MERGE = 0.015

STYLE = {
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.labelsize": 10.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "grid.color": "#d8dee4",
    "grid.linewidth": 0.6,
    "legend.fontsize": 9.5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sweep = json.loads(args.sweep.read_text())
    floor = load_floor(sweep["val_split_hash"])
    by_steps: dict[int, list[dict]] = {}
    for row in sorted(sweep["rows"], key=lambda r: r["guidance"]):
        by_steps.setdefault(row["steps"], []).append(row)

    plt.rcParams.update(STYLE)
    figure, axes = plt.subplot_mosaic(
        [["kid", "manifold"], ["fid", "manifold"]],
        figsize=(9, 5.4), layout="constrained", gridspec_kw={"width_ratios": (1, 1.1)},
    )
    kid_ax, fid_ax, manifold_ax = axes["kid"], axes["fid"], axes["manifold"]
    fid_ax.sharex(kid_ax)
    handles = []

    for index, (color, (steps, rows)) in enumerate(zip(SERIES, sorted(by_steps.items()))):
        guidance = [r["guidance"] for r in rows]
        kid = [1e3 * r["kid"] for r in rows]
        kid_std = [1e3 * r["kid_std"] for r in rows]
        line = dict(color=color, marker="o", markersize=3.5, linewidth=1.8)
        handles.append(Line2D([], [], label=f"{steps} steps", **line))
        kid_ax.fill_between(
            guidance, [k - s for k, s in zip(kid, kid_std)], [k + s for k, s in zip(kid, kid_std)],
            color=color, alpha=0.18, linewidth=0,
        )
        kid_ax.plot(guidance, kid, **line)
        fid_ax.plot(guidance, [r["fid"] for r in rows], **line)

        coverage = [r["coverage"] for r in rows]
        density = [r["density"] for r in rows]
        manifold_ax.plot(coverage, density, **line)
        groups: list[list[tuple[float, float, float]]] = []
        for w, x, y in zip(guidance, coverage, density):
            first = groups[-1][0] if groups else None
            if first and abs(first[1] - x) < LABEL_MERGE and abs(first[2] - y) < LABEL_MERGE:
                groups[-1].append((w, x, y))
            else:
                groups.append([(w, x, y)])
        right = index % 2 == 0
        for group in groups:
            _, x, y = (max if right else min)(group, key=lambda p: p[1])
            manifold_ax.annotate(
                "w=" + ", ".join(f"{w:g}" for w, _, _ in group), (x, y),
                textcoords="offset points", xytext=(7, -3) if right else (-7, 3),
                ha="left" if right else "right", fontsize=8.5, color=INK,
            )

    if floor is not None:
        for ax, value in ((kid_ax, 1e3 * floor["kid"]), (fid_ax, floor["fid"])):
            ax.axhline(value, color=FLOOR_COLOR, linestyle="--", linewidth=1.1)
            ax.annotate(
                "real vs real", (1, value), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(-4, 2), ha="right", va="bottom",
                fontsize=8.5, color=INK,
            )
        manifold_ax.plot(floor["coverage"], floor["density"], marker="*", markersize=12,
                         color=FLOOR_COLOR, linestyle="none")
        handles.append(Line2D([], [], marker="*", markersize=10, color=FLOOR_COLOR, linestyle="none",
                              label=f"real vs real (N = {floor['n']:,})"))

    low = min(*manifold_ax.get_xlim(), *manifold_ax.get_ylim())
    high = max(*manifold_ax.get_xlim(), *manifold_ax.get_ylim())
    manifold_ax.plot([low, high], [low, high], color=FLOOR_COLOR, linestyle=":", linewidth=1.1, zorder=0)
    anchor = low + 0.66 * (high - low)
    manifold_ax.annotate(
        "density = coverage", (anchor, anchor), textcoords="offset points", xytext=(-2, 2),
        va="bottom", fontsize=8.5, color=INK, rotation=45, rotation_mode="anchor",
    )
    manifold_ax.set(xlim=(low, high), ylim=(low, high), aspect="equal")

    kid_ax.set_ylabel("KID ×10⁻³")
    kid_ax.tick_params(labelbottom=False)
    fid_ax.set(xlabel="guidance scale w", ylabel="FID")
    manifold_ax.set(xlabel="coverage (diversity)", ylabel="density (fidelity)")
    for ax in (kid_ax, fid_ax, manifold_ax):
        ax.grid(True)
        ax.set_axisbelow(True)

    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
                  labelcolor=INK)
    out = args.sweep.with_suffix(".png") if args.out is None else args.out
    figure.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(out)


if __name__ == "__main__":
    main()
