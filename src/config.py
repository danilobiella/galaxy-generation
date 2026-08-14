"""Single source of truth for run configuration and filesystem layout."""

from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()

KAGGLE_DATA_PATH = "/kaggle/input/galaxy10-DECaLS"
LOCAL_DATA_PATH = "data/galaxy10_decals_lmdb"

DATA_PATH = KAGGLE_DATA_PATH if ON_KAGGLE else LOCAL_DATA_PATH

OUTPUT_DIR = Path(".") if ON_KAGGLE else Path("output")

FIGURES_DIR = OUTPUT_DIR / "figures"
STATE_PATH = OUTPUT_DIR / "state.pkl"
HISTORY_PATH = OUTPUT_DIR / "history.pkl"
WEIGHTS_PATH = OUTPUT_DIR / "galaxy_unet.safetensors"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_plot.png"
GRADIENT_PLOT_PATH = OUTPUT_DIR / "gradient_plot.png"

IMAGE_SIZE = (64, 64)
TIMESTEPS = 1000

SAMPLER = "ddim"
DDIM_STEPS = 50
DDIM_ETA = 0.0

BATCH_SIZE = 16
N_EPOCHS = 10
SEED = 42
LEARNING_RATE = 1e-4
