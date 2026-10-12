"""FID and KID between two sets of images, on Inception-v3 pool features.

Inception runs through the vendored `fidjax`. Both scores are computed here in
float64 numpy rather than with `fidjax.FID.compute_score`, which takes a float32
matrix square root.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy import linalg

import fidjax
from config import (INCEPTION_WEIGHTS_PATH, INCEPTION_WEIGHTS_SHA256,
                    INCEPTION_WEIGHTS_URL)
from data_loader import GalaxyLMDBDataSource, resize_image

FEATURE_DIM = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inception(path: str | Path = INCEPTION_WEIGHTS_PATH) -> fidjax.FID:
    """The FID Inception network. The weights are a pickle, and unpickling runs
    code, so the file has to match the pinned hash before it is opened."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Inception weights to {path}...")
        urllib.request.urlretrieve(INCEPTION_WEIGHTS_URL, path)

    digest = _sha256(path)
    if digest != INCEPTION_WEIGHTS_SHA256:
        raise ValueError(
            f"{path} has sha256 {digest}, expected {INCEPTION_WEIGHTS_SHA256}. "
            "Refusing to unpickle it."
        )
    return fidjax.FID(str(path))


def inception_features(
    inception: fidjax.FID, images: np.ndarray, batch_size: int = 25
) -> np.ndarray:
    """uint8 images (N, H, W, 3) to (N, 2048) float64 features."""
    if images.dtype != np.uint8:
        raise TypeError(f"Expected uint8 images, got {images.dtype}.")
    feats = [
        np.asarray(inception.compute_acts(jnp.asarray(images[i:i + batch_size])))
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(feats).astype(np.float64)


def fid_score(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    covmean = linalg.sqrtm(sigma_a @ sigma_b)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset) @ (sigma_b + offset))
    covmean = np.real(covmean)

    diff = mu_a - mu_b
    return float(
        diff @ diff + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.trace(covmean)
    )


def _polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x @ y.T / x.shape[1] + 1.0) ** 3


def kid_score(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    n_subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Unbiased MMD² with a cubic polynomial kernel, as mean and std over
    random subsets of both sets."""
    m = min(subset_size, len(feats_a), len(feats_b))
    rng = np.random.default_rng(seed)
    scores = np.empty(n_subsets)
    for i in range(n_subsets):
        a = feats_a[rng.choice(len(feats_a), m, replace=False)]
        b = feats_b[rng.choice(len(feats_b), m, replace=False)]
        k_aa = _polynomial_kernel(a, a)
        k_bb = _polynomial_kernel(b, b)
        k_ab = _polynomial_kernel(a, b)
        within = (k_aa.sum() - np.trace(k_aa) + k_bb.sum() - np.trace(k_bb)) / (
            m * (m - 1)
        )
        scores[i] = within - 2.0 * k_ab.mean()
    return float(scores.mean()), float(scores.std())


def load_images(
    db_path: str, indices: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Real images as uint8 (N, H, W, 3), resized the way training resizes them."""
    source = GalaxyLMDBDataSource(db_path, np.asarray(indices))
    return np.stack(
        [resize_image(source[i]["image"], image_size) for i in range(len(source))]
    )
