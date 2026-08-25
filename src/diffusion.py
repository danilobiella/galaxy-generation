"""Noise schedule and reverse-process samplers, shared by training and inference.

Deliberately free of project-local imports so it stays importable both as
`src.diffusion` (Kaggle layout) and as `diffusion` (editable install).
"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SAMPLERS = ("ddpm", "ddim")


class DiffusionSchedule(NamedTuple):
    T: int
    beta: jnp.ndarray
    alpha: jnp.ndarray
    alpha_hat: jnp.ndarray

    @classmethod
    def create(cls, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        # Linear variance schedule
        beta = jnp.linspace(beta_start, beta_end, T)
        alpha = 1.0 - beta
        alpha_hat = jnp.cumprod(alpha)
        return cls(T=T, beta=beta, alpha=alpha, alpha_hat=alpha_hat)


class Samplers(NamedTuple):
    ddpm_step: Callable
    ddim_step: Callable
    run_ddpm: Callable
    run_ddim: Callable


def check_sampler(name: str) -> str:
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler {name!r}, expected one of {SAMPLERS}.")
    return name


def ddim_timesteps(total: int, n_steps: int) -> np.ndarray:
    """Descending subsequence of `n_steps` timesteps drawn from `range(total)`."""
    if not 1 <= n_steps <= total:
        raise ValueError(f"n_steps must be in [1, {total}], got {n_steps}.")
    return np.linspace(total - 1, 0, n_steps).round().astype(np.int32)


def make_samplers(
    unet, schedule: DiffusionSchedule, image_shape, num_classes: int | None = None
) -> Samplers:
    """Compiled reverse processes for `unet`.

    When `num_classes` is set the samplers expect a `labels` array and a
    `guidance` scale, and every step evaluates the network once on a batch
    holding both the conditional and the unconditional (null-token) inputs.
    `guidance` stays a traced argument, so sweeping it costs no recompilation.
    """

    def predict_noise(params, x, t, labels, guidance):
        t_batch = jnp.full((x.shape[0],), t, dtype=jnp.int32)

        if num_classes is None:
            return unet.apply(params, None, (x, t_batch, None))

        null = jnp.full_like(labels, num_classes)
        both = unet.apply(
            params,
            None,
            (
                jnp.concatenate([x, x]),
                jnp.concatenate([t_batch, t_batch]),
                jnp.concatenate([labels, null]),
            ),
        )
        eps_cond, eps_uncond = jnp.split(both, 2)
        return eps_uncond + guidance * (eps_cond - eps_uncond)

    @jax.jit
    def ddpm_step(params, x, t, key, labels=None, guidance=1.0):
        eps_hat = predict_noise(params, x, t, labels, guidance)

        a_t = schedule.alpha[t]
        a_hat_t = schedule.alpha_hat[t]
        b_t = schedule.beta[t]

        mu = (x - (b_t / jnp.sqrt(1.0 - a_hat_t)) * eps_hat) / jnp.sqrt(a_t)

        z = jax.random.normal(key, x.shape)
        z = jnp.where(t > 0, z, jnp.zeros_like(z))

        return mu + jnp.sqrt(b_t) * z

    @partial(jax.jit, static_argnums=(5,))
    def ddim_step(params, x, t, t_prev, key, clip, eta, labels=None, guidance=1.0):
        eps_hat = predict_noise(params, x, t, labels, guidance)

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
        direction = jnp.sqrt(jnp.maximum(1.0 - a_hat_prev - sigma**2, 0.0)) * eps_hat

        z = jax.random.normal(key, x.shape)

        return jnp.sqrt(a_hat_prev) * x_0 + direction + sigma * z

    @partial(jax.jit, static_argnums=(0,))
    def run_ddpm(n_samples, params, key, labels=None, guidance=1.0):
        init_key, loop_key = jax.random.split(key)
        x_T = jax.random.normal(init_key, (n_samples, *image_shape))

        def body(i, carry):
            x, carry_key = carry
            carry_key, step_key = jax.random.split(carry_key)
            x = ddpm_step(
                params, x, schedule.T - 1 - i, step_key, labels, guidance
            )
            return x, carry_key

        x_0, _ = jax.lax.fori_loop(0, schedule.T, body, (x_T, loop_key))
        return x_0

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def run_ddim(n_samples, n_steps, clip, params, key, eta, labels=None, guidance=1.0):
        ts = jnp.asarray(ddim_timesteps(schedule.T, n_steps))
        ts_prev = jnp.concatenate([ts[1:], jnp.array([-1], dtype=jnp.int32)])

        init_key, loop_key = jax.random.split(key)
        x_T = jax.random.normal(init_key, (n_samples, *image_shape))

        def body(i, carry):
            x, carry_key = carry
            carry_key, step_key = jax.random.split(carry_key)
            x = ddim_step(
                params, x, ts[i], ts_prev[i], step_key, clip, eta, labels, guidance
            )
            return x, carry_key

        x_0, _ = jax.lax.fori_loop(0, n_steps, body, (x_T, loop_key))
        return x_0

    return Samplers(
        ddpm_step=ddpm_step,
        ddim_step=ddim_step,
        run_ddpm=run_ddpm,
        run_ddim=run_ddim,
    )
