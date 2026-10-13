"""FID and KID between two sets of images, on Inception-v3 pool features.

Inception runs through the vendored `fidjax`. Both scores are computed here in
float64 numpy rather than with `fidjax.FID.compute_score`, which takes a float32
matrix square root.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg

from src import fidjax
from src.config import (INCEPTION_WEIGHTS_PATH, INCEPTION_WEIGHTS_SHA256,
                        INCEPTION_WEIGHTS_URL)
from src.data_loader import GalaxyLMDBDataSource, resize_image
from src.generate import to_uint8

def load_inception(path: str | Path = INCEPTION_WEIGHTS_PATH) -> fidjax.FID:
    """The FID Inception network. The weights are a pickle, and unpickling runs
    code, so the file has to match the pinned hash before it is opened."""
    path = Path(path)
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
    """uint8 images (N, H, W, 3) to (N, 2048) float64 features."""
    if images.dtype != np.uint8:
        raise TypeError(f"Expected uint8 images, got {images.dtype}.")
    feats = [
        np.asarray(inception.compute_acts(jnp.asarray(images[i:i + batch_size])))
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(feats).astype(np.float64)


def fid_score(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """tr sqrt(Σa Σb) as the summed square roots of the eigenvalues of the
    symmetric Σa^½ Σb Σa^½, which has the same spectrum."""
    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    w, v = linalg.eigh(sigma_a)
    sqrt_a = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T
    eigvals = linalg.eigvalsh(sqrt_a @ sigma_b @ sqrt_a)
    tr_covmean = np.sqrt(np.clip(eigvals, 0.0, None)).sum()

    diff = mu_a - mu_b
    return float(
        diff @ diff + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * tr_covmean
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
    full_aa = _polynomial_kernel(feats_a, feats_a)
    full_bb = _polynomial_kernel(feats_b, feats_b)
    full_ab = _polynomial_kernel(feats_a, feats_b)
    scores = np.empty(n_subsets)
    for i in range(n_subsets):
        a = rng.choice(len(feats_a), m, replace=False)
        b = rng.choice(len(feats_b), m, replace=False)
        k_aa = full_aa[np.ix_(a, a)]
        k_bb = full_bb[np.ix_(b, b)]
        k_ab = full_ab[np.ix_(a, b)]
        within = (k_aa.sum() - np.trace(k_aa) + k_bb.sum() - np.trace(k_bb)) / (
            m * (m - 1)
        )
        scores[i] = within - 2.0 * k_ab.mean()
    return float(scores.mean()), float(scores.std())


def quality_scores(
    feats: np.ndarray, real_feats: np.ndarray, seed: int = 0
) -> dict[str, float]:
    kid_mean, kid_std = kid_score(feats, real_feats, seed=seed)
    return {"kid": kid_mean, "kid_std": kid_std, "fid": fid_score(feats, real_feats)}


def format_scores(scores: dict[str, float]) -> str:
    return (
        f"KID {scores['kid'] * 1e3:.2f} ± {scores['kid_std'] * 1e3:.2f} ×10⁻³ | "
        f"FID {scores['fid']:.2f}"
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
    real_feats = inception_features(inception, real_images, batch_size)
    n = len(real_images)
    n_batches = -(-n // batch_size)
    labels = None if real_labels is None else np.resize(real_labels, n_batches * batch_size)

    def quality_fn(epoch, train_state):
        start = time.perf_counter()
        generated = np.concatenate([
            to_uint8(samplers.run_ddim(
                batch_size,
                steps,
                True,
                train_state.ema_params,
                jax.random.fold_in(key, i),
                0.0,
                None if labels is None else jnp.asarray(
                    labels[i * batch_size:(i + 1) * batch_size]
                ),
                guidance,
            ))
            for i in range(n_batches)
        ])[:n]
        sampled = time.perf_counter()
        feats = inception_features(inception, generated, batch_size)
        featurized = time.perf_counter()
        scores = quality_scores(feats, real_feats)
        print(
            f"[EPOCH {epoch}] {format_scores(scores)} | sampling {sampled - start:.0f}s, "
            f"inception {featurized - sampled:.0f}s, "
            f"scores {time.perf_counter() - featurized:.0f}s"
        )
        return scores

    return quality_fn
