import pickle
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax  # type: ignore

from src.data_loader import make_data_loaders
from src.models import build_network  # This now needs to return a U-Net
from src.plot import plot_gradients, plot_loss
from src.train import DiffusionSchedule, fit_model, make_model_fns, make_train_fns

BATCH_SIZE = 16
N_EPOCHS = 10
SEED = 42
TIMESTEPS = 1000
DATA_PATH = "data/galaxy10_decals_lmdb"
IMAGE_SIZE = (64, 64)

KAGGLE_DATA_PATH = "/kaggle/input/galaxy10-DECaLS"
LOCAL_DATA_PATH = "data/galaxy10_decals_lmdb"

if Path("/kaggle/input").exists():
    print("Running on Kaggle")
    data_path = KAGGLE_DATA_PATH
else:
    print("Running locally")
    data_path = LOCAL_DATA_PATH

# %% -- data ---------------------------------------------------------------

train_data_loader = make_data_loaders(
    db_path=data_path,
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
    return 1e-4


optimizer = optax.adam(learning_rate=1e-4, b1=0.9, b2=0.999)
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
)

# %% save network state and history ----------------------------------------

with open("state.pkl", "wb") as f:
    pickle.dump(state, f)

with open("history.pkl", "wb") as f:
    pickle.dump(history, f)

# %% -- plots --------------------------------------------------------------

plot_loss(history)
plot_gradients(history)
