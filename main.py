import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import optax  # type: ignore

from src.config import (BATCH_SIZE, CHECKPOINT_EVERY, CKPT_DIR, DATA_PATH,
                        DATA_SEED, DDIM_STEPS, EMA_DECAY, FIGURES_DIR,
                        FINAL_LEARNING_RATE, GRADIENT_PLOT_PATH,
                        GUIDANCE_SCALE, HISTORY_PATH, IMAGE_SIZE,
                        KEEP_CHECKPOINTS, LEARNING_RATE, LOSS_PLOT_PATH,
                        N_EPOCHS, NUM_CLASSES, ON_KAGGLE, P_UNCOND, RESUME,
                        RUNS_DIR, SEED, STATE_PATH, TIMESTEPS,
                        TRAIN_WEIGHTS_PATH, WARMUP_STEPS)
from src import checkpoint
from src.data_loader import make_data_loaders
from src.models import build_network  # This now needs to return a U-Net
from src.plot import plot_gradients, plot_loss
from src.diffusion import DiffusionSchedule, make_samplers
from src.train import fit_model, make_model_fns, make_train_fns
from src.weights import save_params

print("Running on Kaggle" if ON_KAGGLE else "Running locally")

# %% -- data ---------------------------------------------------------------

training_data = make_data_loaders(
    db_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    seed=DATA_SEED,
    image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
)

steps_per_epoch = training_data.steps_per_epoch
n_steps = N_EPOCHS * steps_per_epoch

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {n_steps}")

print("Made data loader")
print(f"Number of training batches: {steps_per_epoch}")
print(f"Number of training samples: {steps_per_epoch * BATCH_SIZE}")

# %% -- model and optimizer setup -------------------------------------------

unet = build_network(num_classes=NUM_CLASSES)
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
)

ckpt_mgr.close()

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
plot_gradients(history, str(GRADIENT_PLOT_PATH))
