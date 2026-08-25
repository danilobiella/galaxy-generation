"""Compare the training histories of two or more runs.

Per-step loss is dominated by which timestep happened to be sampled, so the
step-level curve is far too noisy to read a difference from. Everything here is
aggregated to epoch means, where the timestep noise averages out and a real gap
between runs becomes visible.

Usage:
    python -m scripts.compare_runs
    python -m scripts.compare_runs --runs runs/old runs/new --labels baseline cosine
"""

import argparse
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import N_EPOCHS, RUNS_DIR

DEFAULT_RUNS = (RUNS_DIR / "old", RUNS_DIR / "new")
METRICS = ("loss", "grad_norm", "param_norm")


def load_history(run_dir: Path) -> dict:
    path = run_dir / "history.pkl"
    if not path.exists():
        raise SystemExit(f"No history at {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def by_epoch(values, n_epochs: int) -> np.ndarray:
    """Reshape a per-step series to (n_epochs, steps_per_epoch)."""
    values = np.asarray(values, dtype=np.float64)
    steps_per_epoch = len(values) // n_epochs
    return values[: steps_per_epoch * n_epochs].reshape(n_epochs, steps_per_epoch)


def sign_test(wins: int, n: int) -> float:
    """Two-sided sign test: how surprising is `wins` out of `n` under a coin flip."""
    observed = math.comb(n, wins)
    total = sum(math.comb(n, k) for k in range(n + 1) if math.comb(n, k) <= observed)
    return min(1.0, total / 2**n)


def paired_report(first: str, second: str, a: np.ndarray, b: np.ndarray, tail: int) -> None:
    """Compare two runs epoch by epoch.

    The runs share a seed and a data order, so their epoch means rise and fall
    together and the per-epoch difference is far less noisy than either curve.
    Treating the epochs as independent samples would understate the gap.
    """
    difference = a[-tail:] - b[-tail:]
    wins = int((difference > 0).sum())
    relative = difference.mean() / a[-tail:].mean() * 100

    direction = "lower" if relative > 0 else "higher"
    print(f"\nPaired over the last {tail} epochs:")
    print(f"  {second} loss is {abs(relative):.2f}% {direction} than {first}")
    print(f"  {second} wins {wins}/{tail} epochs   sign test p = {sign_test(max(wins, tail - wins), tail):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--tail", type=int, default=10, help="epochs in the zoom panel")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "figures" / "run_comparison.png")
    args = parser.parse_args()

    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs in length")

    runs = {}
    for label, run_dir in zip(labels, args.runs):
        history = load_history(run_dir)
        runs[label] = {m: by_epoch(history[m], args.epochs) for m in METRICS if m in history}
        print(f"{label:<12} {run_dir}  {len(history['loss']):,} steps  "
              f"{runs[label]['loss'].shape[1]:,} steps/epoch")

    print(f"\n{'run':<12}{'final loss':>18}{'best epoch':>14}{'last-5 mean':>14}"
          f"{'grad norm':>12}{'param norm':>12}")
    print("-" * 82)

    for label, series in runs.items():
        epoch_loss = series["loss"].mean(axis=1)
        final = series["loss"][-1]
        sem = final.std(ddof=1) / np.sqrt(len(final))
        grad = series["grad_norm"].mean(axis=1)[-1] if "grad_norm" in series else float("nan")
        param = series["param_norm"][-1, -1] if "param_norm" in series else float("nan")
        print(f"{label:<12}{final.mean():>12.5f} ±{sem:.5f}"
              f"{epoch_loss.min():>14.5f}{epoch_loss[-5:].mean():>14.5f}"
              f"{grad:>12.5f}{param:>12.2f}")

    if len(runs) == 2:
        first, second = list(runs)
        a = runs[first]["loss"].mean(axis=1)
        b = runs[second]["loss"].mean(axis=1)
        paired_report(first, second, a, b, args.tail)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for label, series in runs.items():
        epochs = np.arange(1, series["loss"].shape[0] + 1)
        epoch_loss = series["loss"].mean(axis=1)

        axes[0][0].plot(epochs, epoch_loss, label=label, linewidth=1.4)
        axes[0][1].plot(epochs[-args.tail:], epoch_loss[-args.tail:], label=label,
                        marker="o", markersize=3, linewidth=1.4)
        if "grad_norm" in series:
            axes[1][0].plot(epochs, series["grad_norm"].mean(axis=1), label=label, linewidth=1.4)
        if "param_norm" in series:
            axes[1][1].plot(epochs, series["param_norm"][:, -1], label=label, linewidth=1.4)

    axes[0][0].set(title="Loss per epoch", xlabel="epoch", ylabel="mean MSE", yscale="log")
    axes[0][1].set(title=f"Loss, last {args.tail} epochs", xlabel="epoch", ylabel="mean MSE")
    axes[1][0].set(title="Gradient norm per epoch", xlabel="epoch", ylabel="mean norm", yscale="log")
    axes[1][1].set(title="Parameter norm", xlabel="epoch", ylabel="norm")

    for axis in axes.flat:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(fontsize=9)

    figure.suptitle("Training run comparison", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"\n  {args.out}")


if __name__ == "__main__":
    main()
