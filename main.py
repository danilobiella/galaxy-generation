import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import optax  # type: ignore

from src.config import (BATCH_SIZE, DATA_PATH, DDIM_STEPS, FIGURES_DIR,
                        FINAL_LEARNING_RATE, GRADIENT_PLOT_PATH,
                        GUIDANCE_SCALE, HISTORY_PATH, IMAGE_SIZE,
                        LEARNING_RATE, LOSS_PLOT_PATH, N_EPOCHS, NUM_CLASSES,
                        ON_KAGGLE, OUTPUT_DIR, P_UNCOND, SEED, STATE_PATH,
                        EMA_DECAY, TIMESTEPS, WARMUP_STEPS, WEIGHTS_PATH)
from src.data_loader import make_data_loaders
from src.models import build_network  # This now needs to return a U-Net
from src.plot import plot_gradients, plot_loss
from src.diffusion import DiffusionSchedule, make_samplers
from src.train import fit_model, make_model_fns, make_train_fns
from src.weights import save_params

print("Running on Kaggle" if ON_KAGGLE else "Running locally")

# %% -- data ---------------------------------------------------------------

train_data_loader = make_data_loaders(
    db_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    seed=SEED,
    image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
)

steps_per_epoch = len(train_data_loader)
n_steps = N_EPOCHS * steps_per_epoch

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {n_steps}")

print("Made data loader")
print(f"Number of training batches: {len(train_data_loader)}")
print(f"Number of training samples: {len(train_data_loader) * BATCH_SIZE}")

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

# %% -- train --------------------------------------------------------------

key = jax.random.PRNGKey(SEED)
state, history = fit_model(
    key=key,
    train_state=train_state,
    train_step_fn=train_step_fn,
    gen_fn=generate_fn,
    lr_schedule=log_lr,
    train_ds=train_data_loader,
    n_epochs=N_EPOCHS,
    figures_dir=str(FIGURES_DIR),
)

# %% save network state and history ----------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(STATE_PATH, "wb") as f:
    pickle.dump(state, f)

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history, f)

save_params(state.model_state.params, WEIGHTS_PATH)
print(f"Saved inference weights to: {WEIGHTS_PATH}")

EMA_WEIGHTS_PATH = WEIGHTS_PATH.with_name(WEIGHTS_PATH.stem + "_ema" + WEIGHTS_PATH.suffix)
save_params(state.ema_params, EMA_WEIGHTS_PATH)
print(f"Saved EMA weights to: {EMA_WEIGHTS_PATH}")

# %% -- plots --------------------------------------------------------------

plot_loss(history, str(LOSS_PLOT_PATH))
plot_gradients(history, str(GRADIENT_PLOT_PATH))
