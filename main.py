import pickle
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax  # type: ignore

from src.data_loader import make_data_loaders
from src.models import build_network
from src.plot import plot_gradients, plot_loss
from src.train import fit_model, make_model_fns, make_train_fns

BATCH_SIZE = 16
N_EPOCHS = 30
SEED = 42
DATA_PATH = "data/galaxy10_decals_lmdb"
IMAGE_SIZE = (256, 256)
LATENT_DIM = 128
N_GEN_STEPS = 1
N_DIS_STEPS = 5

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
    db_path=data_path, batch_size=BATCH_SIZE, seed=SEED
)

steps_per_epoch = len(train_data_loader)
n_steps = N_EPOCHS * steps_per_epoch

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {n_steps}")

print("Made data loader")
print(f"Numer of training batches: {len(train_data_loader)}")
print(f"Number of training samples: {len(train_data_loader) * BATCH_SIZE}")

# %%

gen, dis = build_network()
print("built network")

input = (jnp.zeros((BATCH_SIZE, 256, 256, 3)), jnp.zeros((BATCH_SIZE,)))
print("Discriminator architecture:")
print(
    hk.experimental.tabulate(
        dis, columns=["module", "input", "output", "params_size", "params_bytes"]
    )(input)
)
input = (jnp.zeros((BATCH_SIZE, LATENT_DIM)), jnp.zeros((BATCH_SIZE,)))
print("\nGenerator architecture:")
print(
    hk.experimental.tabulate(
        gen, columns=["module", "input", "output", "params_size", "params_bytes"]
    )(input)
)


def schedule(step):
    return 1e-4

optimizer = optax.adam(learning_rate=1e-4, b1=0.0, b2=0.9)
print("built optimizer and schedule")

# %%

example_batch = next(iter(train_data_loader))
image_shape = example_batch[0].shape

init_model_state, generate_fn = make_model_fns(
    gen, dis, latent_dim=LATENT_DIM, image_shape=image_shape[1:]
)
init_train_state, gen_train_step, dis_train_step = make_train_fns(
    gen, dis, optimizer, latent_dim=LATENT_DIM
)

model_state = init_model_state(rng=jax.random.PRNGKey(SEED))
train_state = init_train_state(model_state)
print("Initialized model and training state")

# %% -- train --------------------------------------------------------------

key = jax.random.PRNGKey(SEED)
state, history = fit_model(
    key=key,
    train_state=train_state,
    gen_step=gen_train_step,
    dis_step=dis_train_step,
    gen_fn=generate_fn,
    schedule=schedule,
    train_ds=train_data_loader,
    n_epochs=N_EPOCHS,
    n_gen_steps=N_GEN_STEPS,
    n_dis_steps=N_DIS_STEPS,
)

# %% save network state and history

with open("state.pkl", "wb") as f:
    pickle.dump(state, f)

with open("history.pkl", "wb") as f:
    pickle.dump(history, f)

# %% -- plots -----------------------------------------------

plot_loss(history)
plot_gradients(history)
