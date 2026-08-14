"""Reusable sampling API for the trained diffusion model."""

import time
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np

from config import DDIM_ETA, DDIM_STEPS, IMAGE_SIZE, SAMPLER, TIMESTEPS, WEIGHTS_PATH
from diffusion import (
    SAMPLERS,
    DiffusionSchedule,
    check_sampler,
    ddim_timesteps,
    make_samplers,
)
from models import build_network
from weights import load_params

__all__ = ["SAMPLERS", "Generator", "load_generator", "random_seed", "to_uint8"]


def to_uint8(images) -> np.ndarray:
    array = np.asarray(jax.device_get(images))
    return np.clip((array + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)


def random_seed() -> int:
    return time.time_ns() % (2**31)


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
        self.samplers = make_samplers(self.unet, self.schedule, self.image_shape)

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
        params=None,
    ) -> np.ndarray:
        """Sample `n_samples` galaxies as a uint8 array of shape (N, H, W, 3)."""
        key = jax.random.PRNGKey(random_seed() if seed is None else seed)
        params = self.params if params is None else params

        if check_sampler(sampler) == "ddpm":
            latents = self.samplers.run_ddpm(n_samples, params, key)
        else:
            latents = self.samplers.run_ddim(n_samples, steps, clip, params, key, eta)

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
        params=None,
    ) -> list[np.ndarray]:
        """Same sampling, but keeping every `stride`-th intermediate state."""
        sampler = check_sampler(sampler)
        pairs = self.step_schedule(sampler, steps)
        params = self.params if params is None else params

        key = jax.random.PRNGKey(random_seed() if seed is None else seed)
        init_key, key = jax.random.split(key)
        x = jax.random.normal(init_key, (n_samples, *self.image_shape))

        frames = []
        for i, (t, t_prev) in enumerate(pairs):
            key, step_key = jax.random.split(key)

            if sampler == "ddpm":
                x = self.samplers.ddpm_step(params, x, t, step_key)
            else:
                x = self.samplers.ddim_step(params, x, t, t_prev, step_key, clip, eta)

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
