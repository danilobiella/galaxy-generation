import math

import matplotlib.pyplot as plt


def plot_loss(history: dict):
    plt.figure(figsize=(8, 4))

    plt.plot(history["dis_loss"], label="dis_loss", color="darkorange", lw=1)
    plt.plot(
        history["dis_real_loss"],
        label="dis_real_loss",
        color="darkorange",
        linestyle="--",
    )
    plt.plot(
        history["dis_fake_loss"],
        label="dis_fake_loss",
        color="darkorange",
        linestyle=":",
    )

    plt.plot(history["gen_loss"], label="gen_loss", color="darkgreen", lw=1)

    plt.axhline(math.log(2), color="red", linestyle="--", label="Equilibrium Loss")

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.savefig("loss_plot.png", dpi=300)


def plot_gradients(history: dict, ax: plt.Axes = None):
    plt.figure(figsize=(8, 4))

    plt.plot(history["dis_grad_norm"], label="dis_grad_norm", color="darkorange")
    plt.plot(history["dis_param_norm"], label="dis_param_norm", color="darkred")
    plt.plot(history["gen_grad_norm"], label="gen_grad_norm", color="darkgreen")
    plt.plot(history["gen_param_norm"], label="gen_param_norm", color="lightgreen")
    plt.legend()
    plt.xlabel("epoch")

    plt.ylabel("norm")
    plt.savefig("gradient_plot.png", dpi=300)
