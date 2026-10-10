"""Single source of truth for run configuration and filesystem layout."""

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
SMOKE = os.environ.get("SMOKE") == "1"

KAGGLE_DATA_PATH = "/kaggle/input/galaxy10-DECaLS"
LOCAL_DATA_PATH = "data/galaxy10_decals_lmdb"

DATA_PATH = KAGGLE_DATA_PATH if ON_KAGGLE else LOCAL_DATA_PATH

RUNS_DIR = Path(".") if ON_KAGGLE else Path("runs")
if SMOKE:
    RUNS_DIR = RUNS_DIR / "smoke"
CHECKPOINTS_DIR = Path("model_checkpoints")
STUDY_DIR = Path("study")

FIGURES_DIR = RUNS_DIR / "figures"
STATE_PATH = RUNS_DIR / "state.pkl"
HISTORY_PATH = RUNS_DIR / "history.pkl"
LOSS_PLOT_PATH = RUNS_DIR / "loss_plot.png"
GRADIENT_PLOT_PATH = RUNS_DIR / "gradient_plot.png"
TRAIN_WEIGHTS_PATH = RUNS_DIR / "galaxy_unet.safetensors"
CKPT_DIR = RUNS_DIR / "ckpt"

WEIGHTS_PATH = CHECKPOINTS_DIR / "galaxy_unet.safetensors"

IMAGE_SIZE = (64, 64)
TIMESTEPS = 1000

SAMPLER = "ddim"
DDIM_STEPS = 50
DDIM_ETA = 0.0

NUM_CLASSES = 10
P_UNCOND = 0.1
GUIDANCE_SCALE = 2.0

CLASS_NAMES = (
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
)

API_STEPS = (20, 50)
API_MAX_BATCH = 10

BATCH_SIZE = 16
N_EPOCHS = 1 if SMOKE else 50
MODEL_DIM = 16 if SMOKE else 64
MODEL_DIM_MULTS = (1, 2, 4) if SMOKE else (1, 2, 4, 8)
SEED = 42
DATA_SEED = 1234
VAL_FRACTION = 0.1
SPLIT_SEED = 0
EVAL_TIMESTEPS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
EMA_DECAY = 0.9999
FINAL_LEARNING_RATE = 1e-6

RESUME = False
CHECKPOINT_EVERY = 5
KEEP_CHECKPOINTS = 2
