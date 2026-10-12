import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from src.diffusion import DiffusionSchedule

T = 20
IMAGE_SHAPE = (8, 8, 3)
NUM_CLASSES = 4


class StubUNet:
    """Stands in for a Haiku-transformed U-Net: only `.apply` is ever called."""

    def __init__(self, eps_fn):
        self.eps_fn = eps_fn
        self.seen_shapes = []

    def apply(self, params, rng, inputs):
        x, t, y = inputs
        self.seen_shapes.append((x.shape, t.shape, None if y is None else y.shape))
        return self.eps_fn(params, x, t, y)


def label_encoding_unet():
    """Returns the label as the predicted noise, so CFG arithmetic is readable."""

    def eps_fn(params, x, t, y):
        return jnp.broadcast_to(y.reshape(-1, 1, 1, 1).astype(x.dtype), x.shape)

    return StubUNet(eps_fn)


def constant_unet(eps):
    def eps_fn(params, x, t, y):
        return jnp.broadcast_to(eps, x.shape)

    return StubUNet(eps_fn)


class TinyNetwork:
    """A learnable stand-in for the U-Net: eps_hat = w * x + b[label].

    Small enough that a training step is instant, but it has real parameters
    and a real gradient, so the optimiser, the EMA and the loss are all
    genuinely exercised.

    `b` has one row per class plus the null token, and only the rows actually
    used get a gradient. That makes `moved_rows` a jit-safe way to see which
    labels reached the network -- recording them in a Python list would only
    ever capture tracers, since `train_step` is compiled.
    """

    def init(self, rng, input):
        return {"w": jnp.array(0.5), "b": jnp.zeros((NUM_CLASSES + 1,))}

    def apply(self, params, rng, input):
        x, t, y = input
        out = params["w"] * x
        if y is not None:
            out = out + params["b"][y].reshape(-1, 1, 1, 1)
        return out


@pytest.fixture
def schedule():
    return DiffusionSchedule.create(T=T)


@pytest.fixture
def sched(schedule):
    """The same schedule; the training and checkpoint tests spell it this way."""
    return schedule


@pytest.fixture
def net():
    return TinyNetwork()


@pytest.fixture
def batch():
    return jax.random.normal(jax.random.key(0), (4, *IMAGE_SHAPE))


@pytest.fixture
def params():
    return {}
