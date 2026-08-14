"""Reusable sampling API for the trained diffusion model."""

import time
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from config import IMAGE_SIZE, TIMESTEPS, WEIGHTS_PATH
from models import build_network
from train import DiffusionSchedule
from weights import load_params


def to_uint8(images) -> np.ndarray:
    array = np.asarray(jax.device_get(images))
    return np.clip((array + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)


def random_seed() -> int:
    return time.time_ns() % (2**31)


class Generator:
    """Holds the weights and the compiled reverse process."""

    def __init__(
        self,
        params,
        image_size: tuple[int, int] = IMAGE_SIZE,
        timesteps: int = TIMESTEPS,
    ):
        self.params = jax.device_put(params)
        self.image_shape = (image_size[0], image_size[1], 3)
        self.schedule = DiffusionSchedule.create(T=timesteps)
        self.unet = build_network()

        schedule = self.schedule
        unet = self.unet
        image_shape = self.image_shape

        @jax.jit
        def reverse_step(params, x, t, key):
            t_batch = jnp.full((x.shape[0],), t, dtype=jnp.int32)
            eps_hat = unet.apply(params, None, (x, t_batch))

            a_t = schedule.alpha[t]
            a_hat_t = schedule.alpha_hat[t]
            b_t = schedule.beta[t]

            mu = (x - (b_t / jnp.sqrt(1.0 - a_hat_t)) * eps_hat) / jnp.sqrt(a_t)

            z = jax.random.normal(key, x.shape)
            z = jnp.where(t > 0, z, jnp.zeros_like(z))

            return mu + jnp.sqrt(b_t) * z

        @partial(jax.jit, static_argnums=(0,))
        def sample(n_samples, params, key):
            init_key, loop_key = jax.random.split(key)
            x_T = jax.random.normal(init_key, (n_samples, *image_shape))

            def body(i, carry):
                x, carry_key = carry
                carry_key, step_key = jax.random.split(carry_key)
                t = schedule.T - 1 - i
                return reverse_step(params, x, t, step_key), carry_key

            x_0, _ = jax.lax.fori_loop(0, schedule.T, body, (x_T, loop_key))
            return x_0

        self._reverse_step = reverse_step
        self._sample = sample

    @classmethod
    def from_weights(
        cls,
        path: str | Path = WEIGHTS_PATH,
        image_size: tuple[int, int] = IMAGE_SIZE,
        timesteps: int = TIMESTEPS,
    ) -> "Generator":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No weights at {path}. Train a model first, or point at a "
                "downloaded galaxy_unet.safetensors."
            )
        return cls(load_params(path), image_size=image_size, timesteps=timesteps)

    def generate(self, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        """Sample `n_samples` galaxies as a uint8 array of shape (N, H, W, 3)."""
        key = jax.random.PRNGKey(random_seed() if seed is None else seed)
        return to_uint8(self._sample(n_samples, self.params, key))

    def generate_trajectory(
        self, n_samples: int = 1, seed: int | None = None, stride: int = 10
    ) -> list[np.ndarray]:
        """Same sampling, but keeping every `stride`-th intermediate state."""
        key = jax.random.PRNGKey(random_seed() if seed is None else seed)
        init_key, key = jax.random.split(key)
        x = jax.random.normal(init_key, (n_samples, *self.image_shape))

        frames = []
        for i in range(self.schedule.T):
            key, step_key = jax.random.split(key)
            x = self._reverse_step(self.params, x, self.schedule.T - 1 - i, step_key)
            if i % stride == 0 or i == self.schedule.T - 1:
                frames.append(to_uint8(x))
        return frames


@lru_cache(maxsize=1)
def _load_cached(path: str, image_size: tuple[int, int], timesteps: int) -> Generator:
    return Generator.from_weights(path, image_size=image_size, timesteps=timesteps)


def load_generator(
    path: str | Path = WEIGHTS_PATH,
    image_size: tuple[int, int] = IMAGE_SIZE,
    timesteps: int = TIMESTEPS,
) -> Generator:
    """Process-wide cached generator, so weights and JIT are paid for once."""
    return _load_cached(str(path), tuple(image_size), timesteps)
