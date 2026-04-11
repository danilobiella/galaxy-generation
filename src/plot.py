import math

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(history: dict):
    plt.figure(figsize=(8, 4))

    history["dis_loss"] = pd.Series(history["dis_loss"]).rolling(window=100).mean()
    history["gen_loss"] = pd.Series(history["gen_loss"]).rolling(window=100).mean()
    history["w_loss"] = pd.Series(history["w_loss"]).rolling(window=100).mean()

    plt.plot(history["dis_loss"], label="dis_loss", color="darkorange")
    plt.plot(
        history["w_loss"],
        label="dis_real_loss",
        color="darkorange",
        linestyle="--",
    )

    plt.plot(history["gen_loss"], label="gen_loss", color="darkgreen")

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.savefig("loss_plot.png", dpi=300)


def plot_gradients(history: dict, ax: plt.Axes = None):
    plt.figure(figsize=(8, 4))

    history["dis_grad_norm"] = (
        pd.Series(history["dis_grad_norm"]).rolling(window=100).mean()
    )
    history["gen_grad_norm"] = (
        pd.Series(history["gen_grad_norm"]).rolling(window=100).mean()
    )

    plt.plot(history["dis_grad_norm"], label="dis_grad_norm", color="darkorange")
    plt.plot(history["gen_grad_norm"], label="gen_grad_norm", color="darkgreen")
    plt.legend()
    plt.xlabel("epoch")

    plt.ylabel("norm")
    plt.savefig("gradient_plot.png", dpi=300)
