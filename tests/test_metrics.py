import pickle
from pathlib import Path

import lmdb
import numpy as np
import pytest
from PIL import Image

from src.config import INCEPTION_WEIGHTS_PATH
from src.data_loader import close_envs, process_features
from src.metrics import (Reference, density_coverage, fid_score,
                         inception_features, kid_score, load_images,
                         load_inception)

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def inception():
    if not INCEPTION_WEIGHTS_PATH.exists():
        pytest.skip(f"No Inception weights at {INCEPTION_WEIGHTS_PATH}")
    return load_inception()


def test_inception_matches_pytorch_fid(inception):
    """Reference output of pytorch-fid's Inception on the same image, from
    jax-fid's test suite (Apache-2.0, github.com/matthias-wright/jax-fid)."""
    image = np.asarray(Image.open(DATA / "elefant.jpg"), dtype=np.float32) / 255.0
    out = inception.apply(image[None] * 2.0 - 1.0)
    target = np.load(DATA / "elefant_inception_output.npy")
    assert np.mean(np.abs(np.asarray(out) - target)) < 1e-5


def test_features_do_not_depend_on_the_batch_size(inception):
    images = np.random.default_rng(0).integers(0, 256, (10, 64, 64, 3), dtype=np.uint8)
    small = inception_features(inception, images, batch_size=3)
    large = inception_features(inception, images, batch_size=7)
    assert small.shape == (10, 2048)
    np.testing.assert_allclose(small, large, atol=1e-4)


def test_fid_is_zero_on_itself_and_the_squared_shift_under_translation():
    feats = np.random.default_rng(0).normal(size=(500, 16))
    shift = np.full(16, 0.5)
    assert fid_score(feats, Reference(feats)) == pytest.approx(0.0, abs=1e-6)
    assert fid_score(feats, Reference(feats + shift)) == pytest.approx(shift @ shift, rel=1e-6)


def test_fid_matches_the_closed_form_for_scaled_covariances():
    """N(0, I) against N(0, s²I) in d dimensions: FID = d (1 - s)²."""
    d, s = 4, 2.0
    a, b = np.random.default_rng(0).normal(size=(2, 100_000, d))
    assert fid_score(a, Reference(s * b)) == pytest.approx(d * (1 - s) ** 2, rel=0.05)


def test_kid_is_zero_within_a_distribution_and_positive_across():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 400, 64))
    same_mean, same_std = kid_score(a, Reference(b), n_subsets=20, subset_size=200)
    shifted_mean, _ = kid_score(a, Reference(b + 1.0), n_subsets=20, subset_size=200)
    assert abs(same_mean) < 3 * same_std
    assert shifted_mean > 10 * same_std


def test_density_and_coverage_separate_fidelity_from_diversity():
    rng = np.random.default_rng(0)
    real, same = rng.normal(size=(2, 1000, 16))
    ref = Reference(real)
    density, coverage = density_coverage(same, ref)
    collapsed = density_coverage(same[:, :1] * np.ones(16) * 0.1, ref)
    assert density == pytest.approx(1.0, abs=0.15)
    assert coverage == pytest.approx(1 - 0.5**5, abs=0.03)
    assert collapsed[1] < 0.2


def test_real_images_are_resized_like_the_training_images(tmp_path):
    images = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    env = lmdb.open(str(tmp_path), map_size=2**22)
    with env.begin(write=True) as txn:
        for i, img in enumerate(images):
            txn.put(f"{i}".encode(), pickle.dumps({"img": img, "label": 0}))
    env.close()

    loaded = load_images(str(tmp_path), np.array([2, 0]), (20, 20))
    close_envs()

    for out, i in zip(loaded, (2, 0)):
        trained, _ = process_features({"image": images[i], "label": 0}, (20, 20))
        np.testing.assert_array_equal(out, np.rint((trained + 1.0) / 2.0 * 255.0))
