import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

WINDOW = 100
TIMESTEP_RAMP = LinearSegmentedColormap.from_list(
    "timestep",
    ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
     "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)


def _smooth(values: list[float], window: int = WINDOW) -> pd.Series:
    """Rolling mean, shrinking the window for runs too short to fill it."""
    series = pd.Series(values)
    window = min(window, max(1, len(series) // 10))
    return series.rolling(window=window, min_periods=1).mean()


def plot_gradients(
    history: dict, filepath: str = "gradient_plot.png", window: int = WINDOW
):
    fig, (ax_grad, ax_param) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    grad_norm = history["grad_norm"]
    ax_grad.plot(grad_norm, color="darkorange", alpha=0.15, linewidth=0.5)
    ax_grad.plot(_smooth(grad_norm, window), label="grad_norm", color="darkorange")
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel("grad norm")
    ax_grad.legend()

    param_norm = history["param_norm"]
    ax_param.plot(_smooth(param_norm, window), label="param_norm", color="steelblue")
    ax_param.set_ylabel("param norm")
    ax_param.set_xlabel("step")
    ax_param.legend()

    fig.tight_layout()
    fig.savefig(filepath, dpi=300)


def plot_loss(history: dict, filepath: str = "loss_plot.png"):
    prefix = "val_mse_t"
    keys = sorted(
        (k for k in history if k.startswith(prefix)), key=lambda k: int(k[len(prefix):])
    )
    val_mse = history["val_mse"]
    epochs = np.arange(1, len(val_mse) + 1)
    train_mse = np.asarray(history["loss"]).reshape(len(val_mse), -1).mean(axis=1)
    colors = TIMESTEP_RAMP(np.linspace(0.0, 1.0, len(keys)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, train_mse, linewidth=2, color="#eb6834", label="train (epoch mean)")
    ax.plot(epochs, val_mse, linewidth=2, color="#2b2b2b", label="val (mean)")
    for key, color in zip(keys, colors):
        ax.plot(
            epochs, history[key], linestyle="--", linewidth=1,
            color=color, label=f"val t = {key[len(prefix):]}",
        )

    ax.set_yscale("log")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (noise prediction)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(filepath, dpi=300)


def plot_sample_quality(history: dict, filepath: str = "sample_quality_plot.png"):
    epochs = np.arange(1, len(history["kid"]) + 1)
    kid = np.asarray(history["kid"]) * 1e3
    kid_std = np.asarray(history["kid_std"]) * 1e3

    fig, (ax_kid, ax_fid) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_kid.fill_between(epochs, kid - kid_std, kid + kid_std, color="#2a78d6", alpha=0.2, linewidth=0)
    ax_kid.plot(epochs, kid, linewidth=2, color="#2a78d6")
    ax_kid.set_ylabel("KID (×10⁻³)")

    ax_fid.plot(epochs, history["fid"], linewidth=2, color="#2b2b2b")
    ax_fid.set_ylabel("FID")
    ax_fid.set_xlabel("epoch")
    ax_fid.xaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in (ax_kid, ax_fid):
        ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
