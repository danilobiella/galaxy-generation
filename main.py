import pickle
import shutil
import sys
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore

from src.config import (BATCH_SIZE, CHECKPOINT_EVERY, CKPT_DIR, DATA_PATH,
                        KAGGLE_STAGED_DATA_PATH,
                        DATA_SEED, DDIM_STEPS, EMA_DECAY, EVAL_TIMESTEPS,
                        FID_BATCH_SIZE, FID_SAMPLES, FID_STEPS,
                        FIGURES_DIR, FINAL_LEARNING_RATE, GRADIENT_PLOT_PATH,
                        GUIDANCE_SCALE, HISTORY_PATH, IMAGE_SIZE,
                        KEEP_CHECKPOINTS, LEARNING_RATE, LOSS_PLOT_PATH,
                        MODEL_DIM, MODEL_DIM_MULTS, N_EPOCHS, NUM_CLASSES,
                        ON_KAGGLE, P_UNCOND, RESUME, RUNS_DIR,
                        SAMPLE_QUALITY_PLOT_PATH, SEED,
                        SPLIT_SEED, STATE_PATH, TIMESTEPS, TRAIN_WEIGHTS_PATH,
                        VAL_FRACTION, WARMUP_STEPS)
from src import checkpoint
from src.data_loader import close_envs, make_data_splits
from src.models import build_network  # This now needs to return a U-Net
from src.plot import plot_gradients, plot_loss, plot_sample_quality
from src.diffusion import DiffusionSchedule, make_samplers
from src.generate import to_uint8
from src.metrics import load_inception, make_sample_quality_fn
from src.train import fit_model, make_eval_fn, make_model_fns, make_train_fns
from src.weights import save_params

sys.stdout.reconfigure(line_buffering=True)
print("Running on Kaggle" if ON_KAGGLE else "Running locally")

# %% -- data ---------------------------------------------------------------

db_path = DATA_PATH
if ON_KAGGLE:
    print(f"Copying {DATA_PATH} to local disk at {KAGGLE_STAGED_DATA_PATH}...")
    shutil.copytree(DATA_PATH, KAGGLE_STAGED_DATA_PATH, dirs_exist_ok=True)
    db_path = KAGGLE_STAGED_DATA_PATH

splits = make_data_splits(
    db_path=db_path,
    batch_size=BATCH_SIZE,
    val_fraction=VAL_FRACTION,
    split_seed=SPLIT_SEED,
    seed=DATA_SEED,
    image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
)
training_data = splits.train

steps_per_epoch = training_data.steps_per_epoch
n_steps = N_EPOCHS * steps_per_epoch

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {n_steps}")

print("Made data loader")
print(f"Number of training batches: {steps_per_epoch}")
print(f"Number of training samples: {steps_per_epoch * BATCH_SIZE}")
print(
    f"Validation: {len(splits.val)} batches, {len(splits.val) * BATCH_SIZE} "
    f"samples | split hash {splits.val_hash}"
)

# %% -- model and optimizer setup -------------------------------------------

unet = build_network(
    num_classes=NUM_CLASSES, dim=MODEL_DIM, dim_mults=MODEL_DIM_MULTS
)
print(f"Built U-Net network (num_classes={NUM_CLASSES})")

# The U-Net takes (images, timesteps)
dummy_images = jnp.zeros((BATCH_SIZE, *IMAGE_SIZE, 3))
dummy_timesteps = jnp.zeros((BATCH_SIZE,), dtype=jnp.int32)
dummy_labels = None if NUM_CLASSES is None else jnp.zeros(
    (BATCH_SIZE,), dtype=jnp.int32
)
unet_input = (dummy_images, dummy_timesteps, dummy_labels)

print("\nU-Net architecture:")
print(
    hk.experimental.tabulate(
        unet, columns=["module", "input", "output", "params_size", "params_bytes"]
    )(unet_input)
)


lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=n_steps,
    end_value=FINAL_LEARNING_RATE,
)


def log_lr(step: int) -> float:
    """The schedule as a plain float, for the training log."""
    return float(lr_schedule(step))


optimizer = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999)
print(
    f"Built optimizer | LR 0 -> {LEARNING_RATE:.1e} over {WARMUP_STEPS} warmup "
    f"steps, cosine decay to {FINAL_LEARNING_RATE:.1e} by step {n_steps}"
)

# %% -- initialize state ---------------------------------------------------

# Initialize the diffusion noise schedule
diffusion_schedule = DiffusionSchedule.create(T=TIMESTEPS)

samplers = make_samplers(
    unet, diffusion_schedule, (*IMAGE_SIZE, 3), num_classes=NUM_CLASSES
)

init_model_state, generate_fn = make_model_fns(
    unet=unet,
    samplers=samplers,
    image_shape=(*IMAGE_SIZE, 3),
    sample_steps=DDIM_STEPS,
    num_classes=NUM_CLASSES,
    guidance=GUIDANCE_SCALE,
)

init_train_state, train_step_fn = make_train_fns(
    unet=unet,
    optimizer=optimizer,
    schedule=diffusion_schedule,
    num_classes=NUM_CLASSES,
    p_uncond=P_UNCOND,
    ema_decay=EMA_DECAY,
)

model_state = init_model_state(rng=jax.random.PRNGKey(SEED))
train_state = init_train_state(model_state)
print("Initialized model and training state")

# %% -- checkpoints --------------------------------------------------------

ckpt_mgr = checkpoint.manager(
    CKPT_DIR, keep=KEEP_CHECKPOINTS, every=CHECKPOINT_EVERY
)

if not RESUME and ckpt_mgr.latest_step() is not None:
    raise SystemExit(
        f"{CKPT_DIR} already holds checkpoints (up to epoch "
        f"{ckpt_mgr.latest_step()}). Move or delete it to start fresh, or set "
        "RESUME = True in src/config.py to continue that run."
    )

meta = checkpoint.run_meta(
    steps_per_epoch=steps_per_epoch,
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    final_learning_rate=FINAL_LEARNING_RATE,
    data_seed=DATA_SEED,
    val_fraction=VAL_FRACTION,
    split_seed=SPLIT_SEED,
    val_split_hash=splits.val_hash,
)

resumed = checkpoint.restore(ckpt_mgr, train_state) if RESUME else None
if RESUME and resumed is None:
    print(f"RESUME is set but {CKPT_DIR} holds no checkpoint; starting fresh")

resumed_history = None
if resumed is not None:
    checkpoint.check_resumable(resumed.meta, meta)
    train_state, resumed_history = resumed.train_state, resumed.history
    print(f"Restored checkpoint at step {int(train_state.step)}")


# %% -- train --------------------------------------------------------------

mse_key, fid_key = jax.random.split(jax.random.PRNGKey(SPLIT_SEED))

mse_fn = make_eval_fn(
    unet=unet,
    schedule=diffusion_schedule,
    val_batches=splits.val,
    key=mse_key,
    timesteps=np.arange(EVAL_TIMESTEPS) * (TIMESTEPS // EVAL_TIMESTEPS)
    + TIMESTEPS // (2 * EVAL_TIMESTEPS),
    num_classes=NUM_CLASSES,
)

print("Loading Inception and featurizing the validation images...")
phase_start = time.perf_counter()
val_images, val_labels = (np.concatenate(a)[:FID_SAMPLES] for a in zip(*splits.val))
quality_fn = make_sample_quality_fn(
    samplers=samplers,
    inception=load_inception(),
    real_images=to_uint8(val_images),
    real_labels=None if NUM_CLASSES is None else val_labels,
    key=fid_key,
    steps=FID_STEPS,
    guidance=GUIDANCE_SCALE,
    batch_size=FID_BATCH_SIZE,
)
print(
    f"FID/KID set up on {len(val_images)} validation images in "
    f"{time.perf_counter() - phase_start:.1f}s"
)


def eval_fn(epoch, train_state):
    return {**mse_fn(epoch, train_state), **quality_fn(epoch, train_state)}


key = jax.random.PRNGKey(SEED)
state, history = fit_model(
    key=key,
    train_state=train_state,
    train_step_fn=train_step_fn,
    gen_fn=generate_fn,
    lr_schedule=log_lr,
    train_ds=training_data.batches,
    steps_per_epoch=steps_per_epoch,
    n_epochs=N_EPOCHS,
    figures_dir=str(FIGURES_DIR),
    history=resumed_history,
    checkpoint_fn=checkpoint.saver(ckpt_mgr, meta),
    eval_fn=eval_fn,
)

ckpt_mgr.close()
close_envs()

# %% save network state and history ----------------------------------------

RUNS_DIR.mkdir(parents=True, exist_ok=True)

with open(STATE_PATH, "wb") as f:
    pickle.dump(state, f)

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history, f)

save_params(state.model_state.params, TRAIN_WEIGHTS_PATH)
print(f"Saved inference weights to: {TRAIN_WEIGHTS_PATH}")

EMA_WEIGHTS_PATH = TRAIN_WEIGHTS_PATH.with_name(
    TRAIN_WEIGHTS_PATH.stem + "_ema" + TRAIN_WEIGHTS_PATH.suffix
)
save_params(state.ema_params, EMA_WEIGHTS_PATH)
print(f"Saved EMA weights to: {EMA_WEIGHTS_PATH}")

# %% -- plots --------------------------------------------------------------

plot_loss(history, str(LOSS_PLOT_PATH))
plot_sample_quality(history, str(SAMPLE_QUALITY_PLOT_PATH))
plot_gradients(history, str(GRADIENT_PLOT_PATH))
