from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config import INCEPTION_WEIGHTS_PATH
from metrics import fid_score, kid_score, load_inception

DATA = Path(__file__).parent / "data"


def test_inception_matches_pytorch_fid():
    """Reference output of pytorch-fid's Inception on the same image, from
    jax-fid's test suite (Apache-2.0, github.com/matthias-wright/jax-fid)."""
    if not INCEPTION_WEIGHTS_PATH.exists():
        pytest.skip(f"No Inception weights at {INCEPTION_WEIGHTS_PATH}")
    image = np.asarray(Image.open(DATA / "elefant.jpg"), dtype=np.float32) / 255.0
    out = load_inception().apply(image[None] * 2.0 - 1.0)
    target = np.load(DATA / "elefant_inception_output.npy")
    assert np.mean(np.abs(np.asarray(out) - target)) < 1e-5


def test_fid_is_zero_on_itself_and_the_squared_shift_under_translation():
    feats = np.random.default_rng(0).normal(size=(500, 16))
    shift = np.full(16, 0.5)
    assert fid_score(feats, feats) == pytest.approx(0.0, abs=1e-6)
    assert fid_score(feats, feats + shift) == pytest.approx(shift @ shift, rel=1e-6)


def test_kid_is_zero_within_a_distribution_and_positive_across():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 400, 64))
    same_mean, same_std = kid_score(a, b, n_subsets=20, subset_size=200)
    shifted_mean, _ = kid_score(a, b + 1.0, n_subsets=20, subset_size=200)
    assert abs(same_mean) < 3 * same_std
    assert shifted_mean > 10 * same_std
