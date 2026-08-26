import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from diffusion import DiffusionSchedule

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


@pytest.fixture
def schedule():
    return DiffusionSchedule.create(T=T)


@pytest.fixture
def params():
    return {}
