"""Visualize a training run: loss/gradient curves plus a sample contact sheet.

Reads history.pkl and figures/ from a run directory and writes the plots back
into it.

Usage:
    python viz_run.py                # visualize output/
    python viz_run.py path/to/run    # visualize an archived run
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402
from plot import WINDOW, plot_gradients, plot_loss  # noqa: E402

DEFAULT_RUN_DIR = OUTPUT_DIR
DIFFUSION_KEYS = {"loss", "grad_norm", "param_norm"}
EPOCH_LINE = re.compile(
    r"\[EPOCH (\d+)\] COMPLETED \| Time: ([\d.]+)s \| AVG Loss: ([\d.]+)"
)


def load_history(run_dir: Path) -> dict:
    path = run_dir / "history.pkl"
    if not path.exists():
        raise SystemExit(f"No history.pkl in {run_dir}")

    with open(path, "rb") as f:
        history = pickle.load(f)

    missing = DIFFUSION_KEYS - set(history)
    if missing:
        raise SystemExit(
            f"{path} has keys {sorted(history)}, which is not a diffusion run.\n"
            "Legacy WGAN runs need the pre-diffusion plotting code (see git history)."
        )
    return history


def epoch_times(run_dir: Path) -> list[float]:
    """Per-epoch wall times, scraped from the Kaggle log if one is present."""
    log = next(iter(run_dir.glob("*.log")), None)
    if log is None:
        return []

    try:
        events = json.loads(log.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []

    times = []
    for event in events:
        if not isinstance(event, dict):
            continue
        match = EPOCH_LINE.search(event.get("data", ""))
        if match:
            times.append(float(match.group(2)))
    return times


def summarize(history: dict, run_dir: Path) -> None:
    loss = np.asarray(history["loss"])
    tail = loss[-WINDOW:]

    print(f"Run: {run_dir}")
    print(f"  steps logged   : {len(loss)}")
    print(f"  final loss     : {tail.mean():.5f}  (mean of last {len(tail)} steps)")
    print(f"  best step loss : {loss.min():.5f} @ step {int(loss.argmin())}")
    print(
        f"  param norm     : {history['param_norm'][0]:.2f}"
        f" -> {history['param_norm'][-1]:.2f}"
    )

    times = epoch_times(run_dir)
    if times:
        print(
            f"  epochs         : {len(times)} @ {np.mean(times):.0f}s avg"
            f" ({sum(times) / 60:.0f} min total)"
        )


def sample_sheet(run_dir: Path, filepath: Path, max_panels: int = 12) -> bool:
    """Contact sheet of per-epoch sample grids, subsampled if there are many."""
    figures = sorted(
        (run_dir / "figures").glob("epoch_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not figures:
        print(f"  (no figures/epoch_*.png under {run_dir}, skipping contact sheet)")
        return False

    if len(figures) > max_panels:
        idx = np.linspace(0, len(figures) - 1, max_panels).round().astype(int)
        figures = [figures[i] for i in idx]

    n_cols = min(3, len(figures))
    n_rows = int(np.ceil(len(figures) / n_cols))

    # Match the panel shape to the saved grids so they are not stretched.
    with Image.open(figures[0]) as probe:
        aspect = probe.width / probe.height
    panel_w = 5.0

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_w * n_cols, (panel_w / aspect) * n_rows)
    )
    flat = np.atleast_1d(axes).ravel()

    for ax in flat:
        ax.set_visible(False)

    for ax, path in zip(flat, figures):
        ax.set_visible(True)
        with Image.open(path) as img:
            ax.imshow(img)
        # Keep the frame (not axis("off")) so adjacent grids stay distinguishable.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.75")
        ax.set_title(f"epoch {path.stem.split('_')[1]}", fontsize=11)

    fig.subplots_adjust(wspace=0.06, hspace=0.2)
    fig.savefig(filepath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the curves and samples of a training run."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=DEFAULT_RUN_DIR,
        type=Path,
        help=f"directory holding history.pkl and figures/ (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW, help="rolling-mean window, in steps"
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    history = load_history(run_dir)
    summarize(history, run_dir)

    written = [run_dir / "loss_plot.png", run_dir / "gradient_plot.png"]
    plot_loss(history, str(written[0]), window=args.window)
    plot_gradients(history, str(written[1]), window=args.window)

    sheet = run_dir / "samples_by_epoch.png"
    if sample_sheet(run_dir, sheet):
        written.append(sheet)

    print("\nWrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
