import pickle

import matplotlib.pyplot as plt
import pandas as pd

from src import plot

FILENAME = "out/new_run/history.pkl"

with open(FILENAME, "rb") as f:
    history = pickle.load(f)

history["gen_loss"] = pd.Series(history["gen_loss"]).rolling(window=100).mean()

history["dis_loss"] = pd.Series(history["dis_loss"]).rolling(window=100).mean()
history["dis_real_loss"] = (
    pd.Series(history["dis_real_loss"]).rolling(window=100).mean()
)
history["dis_fake_loss"] = (
    pd.Series(history["dis_fake_loss"]).rolling(window=100).mean()
)

history["dis_grad_norm"] = (
    pd.Series(history["dis_grad_norm"]).rolling(window=100).mean()
)
history["dis_param_norm"] = (
    pd.Series(history["dis_param_norm"]).rolling(window=100).mean()
)

history["gen_grad_norm"] = (
    pd.Series(history["gen_grad_norm"]).rolling(window=100).mean()
)
history["gen_param_norm"] = (
    pd.Series(history["gen_param_norm"]).rolling(window=100).mean()
)

plot.plot_loss(history)
plot.plot_gradients(history)
