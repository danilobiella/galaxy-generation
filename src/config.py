"""Single source of truth for run configuration and filesystem layout.

Imported as ``src.config`` by main.py and the other src modules (bundler.py
only inlines imports carrying the ``src.`` prefix), and as ``config`` by the
scripts in scripts/, which resolve it through the editable install.
"""

from pathlib import Path

# -- environment -----------------------------------------------------------

ON_KAGGLE = Path("/kaggle/input").exists()

KAGGLE_DATA_PATH = "/kaggle/input/galaxy10-DECaLS"
LOCAL_DATA_PATH = "data/galaxy10_decals_lmdb"

DATA_PATH = KAGGLE_DATA_PATH if ON_KAGGLE else LOCAL_DATA_PATH

# -- filesystem layout -----------------------------------------------------
# A run writes its state, history, plots and figures/ under OUTPUT_DIR. On
# Kaggle that is the working directory, so downloading a finished kernel's
# output into a local output/ reproduces a local run exactly.

OUTPUT_DIR = Path(".") if ON_KAGGLE else Path("output")

FIGURES_DIR = OUTPUT_DIR / "figures"
STATE_PATH = OUTPUT_DIR / "state.pkl"
HISTORY_PATH = OUTPUT_DIR / "history.pkl"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_plot.png"
GRADIENT_PLOT_PATH = OUTPUT_DIR / "gradient_plot.png"

# -- model / diffusion -----------------------------------------------------
# Sampling from a checkpoint requires these to match the values it was
# trained with, so both training and the generation scripts read them here.

IMAGE_SIZE = (64, 64)
TIMESTEPS = 1000

# -- training --------------------------------------------------------------

BATCH_SIZE = 16
N_EPOCHS = 10
SEED = 42
LEARNING_RATE = 1e-4
