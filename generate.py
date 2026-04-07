import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.models import build_network
from src.train import ModelState, TrainState


def plot_images(images):
    plt.figure(figsize=(8, 8))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.imshow((images[i] + 1) / 2)  # Rescale from [-1, 1] to [0, 1]
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

key, subkey = jax.random.split(key)
z = jax.random.normal(subkey, (36, 128))
images, _ = gen.apply(model_state.gen_params, model_state.gen_nn_state, None, z)

plot_images(images)

# %% Truncated samples

TRUNCATION_THRESHOLD = 0.1

key, subkey = jax.random.split(key)
z = jax.random.normal(subkey, (36, 128))
z = jnp.where(jnp.abs(z) > TRUNCATION_THRESHOLD, z, jnp.sign(z) * TRUNCATION_THRESHOLD)
images, _ = gen.apply(model_state.gen_params, model_state.gen_nn_state, None, z)

plot_images(images)

# %% Interpolation

key, subkey = jax.random.split(key)
z1 = jax.random.normal(subkey, (128,))
z1 = jnp.where(jnp.abs(z1) > TRUNCATION_THRESHOLD, z1, jnp.sign(z1) * TRUNCATION_THRESHOLD)
key, subkey = jax.random.split(key)
z2 = jax.random.normal(subkey, (128,))
z2 = jnp.where(jnp.abs(z2) > TRUNCATION_THRESHOLD, z2, jnp.sign(z2) * TRUNCATION_THRESHOLD)
n_interp = 36
z_interp = jnp.array(
    [z1 * (1 - alpha) + z2 * alpha for alpha in jnp.linspace(0, 1, n_interp)]
)

images, _ = gen.apply(model_state.gen_params, model_state.gen_nn_state, None, z_interp)

plot_images(images)

