"""Single source of truth for run configuration and filesystem layout."""

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
SMOKE = os.environ.get("SMOKE") == "1"

KAGGLE_DATA_PATH = "/kaggle/input/galaxy10-DECaLS"
LOCAL_DATA_PATH = "data/galaxy10_decals_lmdb"
KAGGLE_STAGED_DATA_PATH = "/tmp/galaxy10_decals_lmdb"

DATA_PATH = KAGGLE_DATA_PATH if ON_KAGGLE else LOCAL_DATA_PATH

RUNS_DIR = Path(".") if ON_KAGGLE else Path("runs")
if SMOKE:
    RUNS_DIR = RUNS_DIR / "smoke"
CHECKPOINTS_DIR = Path("model_checkpoints")
STUDY_DIR = Path("study")
RESULTS_DIR = Path("results")
QUALITY_BASELINE_PATH = RESULTS_DIR / "quality_baseline.json"

FIGURES_DIR = RUNS_DIR / "figures"
STATE_PATH = RUNS_DIR / "state.pkl"
HISTORY_PATH = RUNS_DIR / "history.pkl"
EPOCHS_TABLE_PATH = RUNS_DIR / "epochs.csv"
LOSS_PLOT_PATH = RUNS_DIR / "loss_plot.png"
SAMPLE_QUALITY_PLOT_PATH = RUNS_DIR / "sample_quality_plot.png"
GRADIENT_PLOT_PATH = RUNS_DIR / "gradient_plot.png"
TRAIN_WEIGHTS_PATH = RUNS_DIR / "galaxy_unet.safetensors"
CKPT_DIR = RUNS_DIR / "ckpt"

WEIGHTS_PATH = CHECKPOINTS_DIR / "galaxy_unet_ema.safetensors"
WEIGHTS_URL = (
    "https://huggingface.co/danilobiella/galaxy-generation/resolve/main/"
    "galaxy_unet_ema.safetensors"
)
WEIGHTS_SHA256 = (
    "f9384f919b4da2539483f0b19957d818bf3e50d4d29513ca7cc578e372ec654b"
)
INCEPTION_WEIGHTS_PATH = CHECKPOINTS_DIR / "inception_v3_weights_fid.pickle"
INCEPTION_WEIGHTS_URL = (
    "https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1"
)
INCEPTION_WEIGHTS_SHA256 = (
    "4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77"
)

IMAGE_SIZE = (64, 64)
TIMESTEPS = 1000

SAMPLER = "ddim"
DDIM_STEPS = 50
DDIM_ETA = 0.0

NUM_CLASSES = 10
P_UNCOND = 0.1
GUIDANCE_SCALE = 1.0

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
N_EPOCHS = 50
STEPS_PER_EPOCH = None
PRINT_EVERY = 100
EVAL_BATCHES = None
MODEL_DIM = 64
MODEL_DIM_MULTS = (1, 2, 4, 8)
SEED = 42
DATA_SEED = 1234
VAL_FRACTION = 0.1
SPLIT_SEED = 0
EVAL_TIMESTEPS = 10
FID_SAMPLES = None
FID_EVERY = 5
FID_STEPS = 20
FID_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
EMA_DECAY = 0.9999
FINAL_LEARNING_RATE = 1e-6

if SMOKE:
    N_EPOCHS, STEPS_PER_EPOCH, EVAL_BATCHES, PRINT_EVERY = 4, 50, 4, 10
    MODEL_DIM, MODEL_DIM_MULTS = 16, (1, 2, 4)
    FID_SAMPLES, FID_EVERY, FID_BATCH_SIZE = 64, 2, 32
    WARMUP_STEPS = 10

RESUME = False
CHECKPOINT_EVERY = 5
KEEP_CHECKPOINTS = 2
