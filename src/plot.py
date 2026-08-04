import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 100


def _smooth(values: list[float], window: int = WINDOW) -> pd.Series:
    """Rolling mean, shrinking the window for runs too short to fill it."""
    series = pd.Series(values)
    window = min(window, max(1, len(series) // 10))
    return series.rolling(window=window, min_periods=1).mean()


def plot_loss(history: dict, filepath: str = "loss_plot.png", window: int = WINDOW):
    plt.figure(figsize=(8, 4))

    loss = history["loss"]
    plt.plot(loss, color="darkgreen", alpha=0.15, linewidth=0.5)
    plt.plot(_smooth(loss, window), label="loss", color="darkgreen")

    plt.yscale("log")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("MSE (noise prediction)")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)


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
