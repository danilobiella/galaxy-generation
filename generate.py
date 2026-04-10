import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.models import build_network
from src.train import ModelState, TrainState


def plot_images(images):
    """
    Plot images, dynamically adjust plot grid so it's a square as possible
    """
    images = (images + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
    plt.figure(figsize=(8, 8))
    n_images = images.shape[0]
    n_cols = int(jnp.ceil(jnp.sqrt(n_images)))
    n_rows = int(jnp.ceil(n_images / n_cols))
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow((images[i] * 255).astype(jnp.uint8))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("generated_images.png")
    plt.show()


with open("out/state.pkl", "rb") as f:
    state: TrainState = pickle.load(f)

model_state: ModelState = state.model_state

gen, _ = build_network()
key = jax.random.PRNGKey(0)

# %% Normal samples

CLASS = 0.0

key, subkey = jax.random.split(key)
z = jax.random.normal(subkey, (8, 128))
input = (z, jnp.full((8,), CLASS))
images, _ = gen.apply(model_state.gen_params, model_state.gen_nn_state, None, input)

plot_images(images)

# %% Truncated samples

TRUNCATION_THRESHOLD = 0.1

key, subkey = jax.random.split(key)
z = jax.random.normal(subkey, (8, 128))
z = jnp.where(jnp.abs(z) > TRUNCATION_THRESHOLD, z, jnp.sign(z) * TRUNCATION_THRESHOLD)
c = jnp.ones((8,)) * CLASS  # No conditioning
input = (z, c)
images, _ = gen.apply(model_state.gen_params, model_state.gen_nn_state, None, input)

plot_images(images)

# %% Interpolation

key, subkey = jax.random.split(key)
z1 = jax.random.normal(subkey, (128,))
z1 = jnp.where(
    jnp.abs(z1) > TRUNCATION_THRESHOLD, z1, jnp.sign(z1) * TRUNCATION_THRESHOLD
)
key, subkey = jax.random.split(key)
z2 = jax.random.normal(subkey, (128,))
z2 = jnp.where(
    jnp.abs(z2) > TRUNCATION_THRESHOLD, z2, jnp.sign(z2) * TRUNCATION_THRESHOLD
)
n_interp = 36
z_interp = jnp.array(
    [z1 * (1 - alpha) + z2 * alpha for alpha in jnp.linspace(0, 1, n_interp)]
)

images, _ = gen.apply(model_state.gen_params, model_state.gen_nn_state, None, z_interp)

plot_images(images)
