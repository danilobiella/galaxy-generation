import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import optax  # type: ignore

from src.config import (BATCH_SIZE, DATA_PATH, FIGURES_DIR, GRADIENT_PLOT_PATH,
                        HISTORY_PATH, IMAGE_SIZE, LEARNING_RATE, LOSS_PLOT_PATH,
                        N_EPOCHS, ON_KAGGLE, OUTPUT_DIR, SEED, STATE_PATH,
                        TIMESTEPS, WEIGHTS_PATH)
from src.data_loader import make_data_loaders
from src.models import build_network  # This now needs to return a U-Net
from src.plot import plot_gradients, plot_loss
from src.diffusion import DiffusionSchedule
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

unet = build_network()
print("Built U-Net network")

# The U-Net takes (images, timesteps)
dummy_images = jnp.zeros((BATCH_SIZE, *IMAGE_SIZE, 3))
dummy_timesteps = jnp.zeros((BATCH_SIZE,), dtype=jnp.int32)
unet_input = (dummy_images, dummy_timesteps)

print("\nU-Net architecture:")
print(
    hk.experimental.tabulate(
        unet, columns=["module", "input", "output", "params_size", "params_bytes"]
    )(unet_input)
)


def lr_schedule(step):
    # You can swap this for optax.warmup_cosine_decay_schedule if needed
    return LEARNING_RATE


optimizer = optax.adam(learning_rate=LEARNING_RATE, b1=0.9, b2=0.999)
print("Built optimizer and schedule")

# %% -- initialize state ---------------------------------------------------

# Initialize the diffusion noise schedule
diffusion_schedule = DiffusionSchedule.create(T=TIMESTEPS)

init_model_state, generate_fn = make_model_fns(
    unet=unet, schedule=diffusion_schedule, image_shape=(*IMAGE_SIZE, 3)
)

init_train_state, train_step_fn = make_train_fns(
    unet=unet, optimizer=optimizer, schedule=diffusion_schedule
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
    lr_schedule=lr_schedule,
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

# %% -- plots --------------------------------------------------------------

plot_loss(history, str(LOSS_PLOT_PATH))
plot_gradients(history, str(GRADIENT_PLOT_PATH))
