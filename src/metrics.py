"""FID and KID between two sets of images, on Inception-v3 pool features.

Inception runs through the vendored `fidjax`. Both scores are computed here in
float64 numpy rather than with `fidjax.FID.compute_score`, which takes a float32
matrix square root.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg

from src import fidjax
from src.config import (FID_FLOOR_PATH, INCEPTION_WEIGHTS_PATH,
                        INCEPTION_WEIGHTS_SHA256, INCEPTION_WEIGHTS_URL)
from src.data_loader import GalaxyLMDBDataSource, resize_image
from src.generate import sample_batched
from src.logs import format_duration


def load_floor(val_hash: str | None = None) -> dict | None:
    """The saved real-vs-real scores from `scripts.fid_floor`, or None if they
    are missing or were computed on a different validation split."""
    if not FID_FLOOR_PATH.exists():
        return None
    floor = json.loads(FID_FLOOR_PATH.read_text())
    if val_hash is not None and floor["val_split_hash"] != val_hash:
        return None
    return floor


def load_inception() -> fidjax.FID:
    """The FID Inception network. The weights are a pickle, and unpickling runs
    code, so the file has to match the pinned hash before it is opened."""
    path = INCEPTION_WEIGHTS_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Inception weights to {path}...")
        urllib.request.urlretrieve(INCEPTION_WEIGHTS_URL, path)

    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != INCEPTION_WEIGHTS_SHA256:
        raise ValueError(
            f"{path} has sha256 {digest}, expected {INCEPTION_WEIGHTS_SHA256}. "
            "Refusing to unpickle it."
        )
    return fidjax.FID(str(path))


def inception_features(
    inception: fidjax.FID, images: np.ndarray, batch_size: int = 25
) -> np.ndarray:
    """uint8 images (N, H, W, 3) to (N, 2048) float64 features. The last batch
    is padded to full size so Inception compiles once."""
    if images.dtype != np.uint8:
        raise TypeError(f"Expected uint8 images, got {images.dtype}.")
    n = len(images)
    pad = -n % batch_size
    if pad:
        images = np.concatenate([images, np.repeat(images[-1:], pad, axis=0)])
    feats = [
        inception.compute_acts(jnp.asarray(images[i:i + batch_size]))
        for i in range(0, len(images), batch_size)
    ]
    return np.asarray(jnp.concatenate(feats))[:n].astype(np.float64)


NEAREST_K = 5


def _polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x @ y.T / x.shape[1] + 1.0) ** 3


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2.0 * x @ y.T
    return np.maximum(d, 0.0)


class Reference:
    """Statistics of a fixed set of real features, computed once and shared by
    every score against it."""

    def __init__(self, feats: np.ndarray):
        self.feats = feats
        self.mu = feats.mean(axis=0)
        self.sigma = np.cov(feats, rowvar=False)

    @cached_property
    def sqrt_sigma(self) -> np.ndarray:
        w, v = linalg.eigh(self.sigma)
        return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T

    @cached_property
    def kernel(self) -> np.ndarray:
        return _polynomial_kernel(self.feats, self.feats)

    @cached_property
    def knn_radii(self) -> np.ndarray:
        """Squared distance from each feature to its `NEAREST_K`-th nearest
        neighbour among the others."""
        d = _squared_distances(self.feats, self.feats)
        return np.partition(d, NEAREST_K, axis=1)[:, NEAREST_K]


def fid_score(feats: np.ndarray, ref: Reference) -> float:
    sigma = np.cov(feats, rowvar=False)
    eigvals = linalg.eigvalsh(ref.sqrt_sigma @ sigma @ ref.sqrt_sigma)
    tr_covmean = np.sqrt(np.clip(eigvals, 0.0, None)).sum()

    diff = feats.mean(axis=0) - ref.mu
    return float(
        diff @ diff + np.trace(sigma) + np.trace(ref.sigma) - 2.0 * tr_covmean
    )


def kid_score(
    feats: np.ndarray,
    ref: Reference,
    n_subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Unbiased MMD² with a cubic polynomial kernel, as mean and std over
    random subsets of both sets."""
    m = min(subset_size, len(feats), len(ref.feats))
    rng = np.random.default_rng(seed)
    full_aa = _polynomial_kernel(feats, feats)
    full_ab = _polynomial_kernel(feats, ref.feats)
    scores = np.empty(n_subsets)
    for i in range(n_subsets):
        a = rng.choice(len(feats), m, replace=False)
        b = rng.choice(len(ref.feats), m, replace=False)
        k_aa = full_aa[np.ix_(a, a)]
        k_bb = ref.kernel[np.ix_(b, b)]
        k_ab = full_ab[np.ix_(a, b)]
        within = (k_aa.sum() - np.trace(k_aa) + k_bb.sum() - np.trace(k_bb)) / (
            m * (m - 1)
        )
        scores[i] = within - 2.0 * k_ab.mean()
    return float(scores.mean()), float(scores.std())


def density_coverage(feats: np.ndarray, ref: Reference) -> tuple[float, float]:
    """Density and coverage (Naeem et al. 2020) over k-NN balls around the real
    features. Density is fidelity: how many real balls a sample falls in, ~1
    for real data. Coverage is diversity: the fraction of real balls holding at
    least one sample."""
    inside = _squared_distances(feats, ref.feats) < ref.knn_radii[None, :]
    density = inside.sum() / (NEAREST_K * len(feats))
    return float(density), float(inside.any(axis=0).mean())


def quality_scores(feats: np.ndarray, ref: Reference, seed: int = 0) -> dict[str, float]:
    kid_mean, kid_std = kid_score(feats, ref, seed=seed)
    density, coverage = density_coverage(feats, ref)
    return {
        "kid": kid_mean,
        "kid_std": kid_std,
        "fid": fid_score(feats, ref),
        "density": density,
        "coverage": coverage,
    }


def format_scores(scores: dict[str, float]) -> str:
    return (
        f"KID {scores['kid'] * 1e3:.2f} ± {scores['kid_std'] * 1e3:.2f} ×10⁻³ | "
        f"FID {scores['fid']:.2f} | "
        f"density {scores['density']:.3f} | coverage {scores['coverage']:.3f}"
    )


def load_images(
    db_path: str, indices: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Real images as uint8 (N, H, W, 3), resized the way training resizes them."""
    source = GalaxyLMDBDataSource(db_path, np.asarray(indices))
    return np.stack(
        [resize_image(source[i]["image"], image_size) for i in range(len(source))]
    )


def make_sample_quality_fn(
    samplers,
    inception: fidjax.FID,
    real_images: np.ndarray,
    real_labels: np.ndarray | None,
    key: jax.Array,
    steps: int,
    guidance: float,
    batch_size: int,
):
    """Per-epoch FID and KID of EMA samples against `real_images`.

    One sample per real image, with that image's label, so the class mix
    matches. The noise is the same every epoch.
    """
    ref = Reference(inception_features(inception, real_images, batch_size))
    n = len(real_images)

    def quality_fn(epoch, train_state):
        start = time.perf_counter()
        generated = sample_batched(
            samplers, train_state.ema_params, key, n, batch_size, steps,
            real_labels, guidance,
        )
        sampled = time.perf_counter()
        feats = inception_features(inception, generated, batch_size)
        featurized = time.perf_counter()
        scores = quality_scores(feats, ref)
        print(
            f"[EPOCH {epoch}] {format_scores(scores)} | "
            f"sampling {format_duration(sampled - start)}, "
            f"inception {format_duration(featurized - sampled)}, "
            f"scores {format_duration(time.perf_counter() - featurized)}"
        )
        return scores

    return quality_fn
