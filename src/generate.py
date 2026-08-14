"""Reusable sampling API for the trained diffusion model."""

import time
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from config import DDIM_ETA, DDIM_STEPS, IMAGE_SIZE, SAMPLER, TIMESTEPS, WEIGHTS_PATH
from models import build_network
from train import DiffusionSchedule
from weights import load_params

SAMPLERS = ("ddpm", "ddim")


def to_uint8(images) -> np.ndarray:
    array = np.asarray(jax.device_get(images))
    return np.clip((array + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)


def random_seed() -> int:
    return time.time_ns() % (2**31)


def check_sampler(name: str) -> str:
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler {name!r}, expected one of {SAMPLERS}.")
    return name


def ddim_timesteps(total: int, n_steps: int) -> np.ndarray:
    """Descending subsequence of `n_steps` timesteps drawn from `range(total)`."""
    if not 1 <= n_steps <= total:
        raise ValueError(f"n_steps must be in [1, {total}], got {n_steps}.")
    return np.linspace(total - 1, 0, n_steps).round().astype(np.int32)


class Generator:
    """Holds the weights and the compiled reverse processes."""

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

        def predict_noise(params, x, t):
            t_batch = jnp.full((x.shape[0],), t, dtype=jnp.int32)
            return unet.apply(params, None, (x, t_batch))

        @jax.jit
        def ddpm_step(params, x, t, key):
            eps_hat = predict_noise(params, x, t)

            a_t = schedule.alpha[t]
            a_hat_t = schedule.alpha_hat[t]
            b_t = schedule.beta[t]

            mu = (x - (b_t / jnp.sqrt(1.0 - a_hat_t)) * eps_hat) / jnp.sqrt(a_t)

            z = jax.random.normal(key, x.shape)
            z = jnp.where(t > 0, z, jnp.zeros_like(z))

            return mu + jnp.sqrt(b_t) * z

        @partial(jax.jit, static_argnums=(5,))
        def ddim_step(params, x, t, t_prev, key, clip, eta):
            eps_hat = predict_noise(params, x, t)

            a_hat_t = schedule.alpha_hat[t]
            a_hat_prev = jnp.where(
                t_prev >= 0, schedule.alpha_hat[jnp.maximum(t_prev, 0)], 1.0
            )

            x_0 = (x - jnp.sqrt(1.0 - a_hat_t) * eps_hat) / jnp.sqrt(a_hat_t)
            if clip:
                x_0 = jnp.clip(x_0, -1.0, 1.0)

            sigma = (
                eta
                * jnp.sqrt((1.0 - a_hat_prev) / (1.0 - a_hat_t))
                * jnp.sqrt(1.0 - a_hat_t / a_hat_prev)
            )
            direction = (
                jnp.sqrt(jnp.maximum(1.0 - a_hat_prev - sigma**2, 0.0)) * eps_hat
            )

            z = jax.random.normal(key, x.shape)

            return jnp.sqrt(a_hat_prev) * x_0 + direction + sigma * z

        @partial(jax.jit, static_argnums=(0,))
        def run_ddpm(n_samples, params, key):
            init_key, loop_key = jax.random.split(key)
            x_T = jax.random.normal(init_key, (n_samples, *image_shape))

            def body(i, carry):
                x, carry_key = carry
                carry_key, step_key = jax.random.split(carry_key)
                return ddpm_step(params, x, schedule.T - 1 - i, step_key), carry_key

            x_0, _ = jax.lax.fori_loop(0, schedule.T, body, (x_T, loop_key))
            return x_0

        @partial(jax.jit, static_argnums=(0, 1, 2))
        def run_ddim(n_samples, n_steps, clip, params, key, eta):
            ts = jnp.asarray(ddim_timesteps(schedule.T, n_steps))
            ts_prev = jnp.concatenate([ts[1:], jnp.array([-1], dtype=jnp.int32)])

            init_key, loop_key = jax.random.split(key)
            x_T = jax.random.normal(init_key, (n_samples, *image_shape))

            def body(i, carry):
                x, carry_key = carry
                carry_key, step_key = jax.random.split(carry_key)
                x = ddim_step(params, x, ts[i], ts_prev[i], step_key, clip, eta)
                return x, carry_key

            x_0, _ = jax.lax.fori_loop(0, n_steps, body, (x_T, loop_key))
            return x_0

        self._ddpm_step = ddpm_step
        self._ddim_step = ddim_step
        self._run_ddpm = run_ddpm
        self._run_ddim = run_ddim

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

    def step_schedule(self, sampler: str, steps: int) -> list[tuple[int, int]]:
        """The (t, t_prev) pairs a sampler walks through, newest noise first."""
        if check_sampler(sampler) == "ddpm":
            return [(t, t - 1) for t in range(self.schedule.T - 1, -1, -1)]

        ts = ddim_timesteps(self.schedule.T, steps)
        return [(int(t), int(p)) for t, p in zip(ts, np.append(ts[1:], -1))]

    def generate(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        sampler: str = SAMPLER,
        steps: int = DDIM_STEPS,
        eta: float = DDIM_ETA,
        clip: bool = True,
    ) -> np.ndarray:
        """Sample `n_samples` galaxies as a uint8 array of shape (N, H, W, 3)."""
        key = jax.random.PRNGKey(random_seed() if seed is None else seed)

        if check_sampler(sampler) == "ddpm":
            latents = self._run_ddpm(n_samples, self.params, key)
        else:
            latents = self._run_ddim(n_samples, steps, clip, self.params, key, eta)

        return to_uint8(latents)

    def generate_trajectory(
        self,
        n_samples: int = 1,
        seed: int | None = None,
        sampler: str = SAMPLER,
        steps: int = DDIM_STEPS,
        eta: float = DDIM_ETA,
        clip: bool = True,
        stride: int = 1,
    ) -> list[np.ndarray]:
        """Same sampling, but keeping every `stride`-th intermediate state."""
        sampler = check_sampler(sampler)
        pairs = self.step_schedule(sampler, steps)

        key = jax.random.PRNGKey(random_seed() if seed is None else seed)
        init_key, key = jax.random.split(key)
        x = jax.random.normal(init_key, (n_samples, *self.image_shape))

        frames = []
        for i, (t, t_prev) in enumerate(pairs):
            key, step_key = jax.random.split(key)

            if sampler == "ddpm":
                x = self._ddpm_step(self.params, x, t, step_key)
            else:
                x = self._ddim_step(self.params, x, t, t_prev, step_key, clip, eta)

            if i % stride == 0 or i == len(pairs) - 1:
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
