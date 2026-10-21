"""Plot a training run: denoising loss, KID and FID per epoch, with the
real-vs-real floor in the quality panels.

Usage:
    python -m scripts.plot_training_curves runs/good_run --out assets/training_curves.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from scripts.plot_sampler_metrics import FLOOR_COLOR, INK, STYLE
from src.config import RUNS_DIR
from src.metrics import load_floor

COLOR = "darkorange"


def label_floor(ax, value: float) -> None:
    ax.axhline(value, color=FLOOR_COLOR, linestyle="--", linewidth=1.1)
    ax.annotate(
        "real vs real", (1, value), xycoords=("axes fraction", "data"),
        textcoords="offset points", xytext=(-4, 3), ha="right", va="bottom",
        fontsize=10, color=INK,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, nargs="?", default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with open(args.run_dir / "history.pkl", "rb") as f:
        table = pd.DataFrame(pickle.load(f)["epochs"]).set_index("epoch")
    scored = table.dropna(subset=["kid"])
    floor = load_floor()

    plt.rcParams.update(STYLE)
    figure, (loss_ax, kid_ax, fid_ax) = plt.subplots(
        1, 3, figsize=(8.4, 3.0), sharex=True, layout="constrained"
    )
    figure.get_layout_engine().set(wspace=0.04)

    loss_ax.plot(table.index, table["train_loss"], color=COLOR, linewidth=1.2, linestyle="--")
    loss_ax.plot(table.index, table["val_mse"], color=COLOR, linewidth=1.8)
    loss_ax.annotate("train", (table.index[-1], table["train_loss"].iloc[-1]),
                     textcoords="offset points", xytext=(0, 12), ha="right", fontsize=10, color=INK)
    loss_ax.annotate("validation", (table.index[-1], table["val_mse"].iloc[-1]),
                     textcoords="offset points", xytext=(0, 6), ha="right", fontsize=10, color=INK)
    loss_ax.set(yscale="log", ylabel="denoising MSE")

    kid = 1e3 * scored["kid"]
    kid_std = 1e3 * scored["kid_std"]
    kid_ax.fill_between(scored.index, kid - kid_std, kid + kid_std, color=COLOR, alpha=0.18, linewidth=0)
    kid_ax.plot(scored.index, kid, color=COLOR, marker="o", markersize=3.5, linewidth=1.8)
    label_floor(kid_ax, 1e3 * floor["kid"])
    kid_ax.set(yscale="log", ylabel="KID ×10⁻³")

    fid_ax.plot(scored.index, scored["fid"], color=COLOR, marker="o", markersize=3.5, linewidth=1.8)
    label_floor(fid_ax, floor["fid"])
    fid_ax.set(yscale="log", ylabel="FID")

    for ax in (loss_ax, kid_ax, fid_ax):
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.grid(True, which="major")
        ax.set_axisbelow(True)

    figure.suptitle("KID and FID: guidance w = 2, 20 DDIM steps",
                    fontsize=10, color=INK, x=0.99, ha="right")
    out = args.run_dir / "training_curves.png" if args.out is None else args.out
    figure.savefig(out, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(figure)
    print(out)


if __name__ == "__main__":
    main()
