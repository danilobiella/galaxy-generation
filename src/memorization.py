"""Nearest-neighbour search between generated samples and the training set.

Answers "is the model copying its training data?". A nearest-neighbour distance
is meaningless on its own, so every report is paired with a baseline: the same
search run with *real* galaxies as queries. If generated samples sit at the same
distance from the training set as other real galaxies do, there is no evidence
of memorisation.
"""

import pickle
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image

METRICS = ("l2", "corr")
INK = "#8b949e"


def load_dataset(
    db_path: str | Path,
    image_size: tuple[int, int],
    limit: int | None = None,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Every training image, resized to `image_size`, as float32 in [0, 1]."""
    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path)

    env = lmdb.open(str(db_path), readonly=True, lock=False)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        total = meta["num_samples"] if limit is None else min(limit, meta["num_samples"])

        height, width = image_size
        images = np.empty((total, height, width, 3), dtype=np.float32)

        for i in range(total):
            sample = pickle.loads(txn.get(f"{i}".encode()))
            pil = Image.fromarray(sample["img"])
            if pil.size != (width, height):
                pil = pil.resize((width, height), Image.Resampling.BILINEAR)
            images[i] = np.asarray(pil, dtype=np.float32) / 255.0
    env.close()

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, images)

    return images


def load_labels(db_path: str | Path, limit: int | None = None) -> np.ndarray:
    """Class label for every training image, in the same order as `load_dataset`."""
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        total = meta["num_samples"] if limit is None else min(limit, meta["num_samples"])
        labels = np.empty(total, dtype=np.int32)
        for i in range(total):
            labels[i] = pickle.loads(txn.get(f"{i}".encode()))["label"]
    env.close()
    return labels


def to_float(images: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] or float [0, 1] in, float32 [0, 1] out."""
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    return images.astype(np.float32, copy=False)


def dihedral_variants(images: np.ndarray) -> np.ndarray:
    """The 8 rotations/reflections of each image, stacked on a leading axis.

    A galaxy's orientation on the sky is arbitrary, so a generated sample that
    is a rotated copy of a training image is still a copy. Plain pixel distance
    would miss it entirely.
    """
    variants = []
    for flip in (False, True):
        base = images[:, :, ::-1] if flip else images
        for k in range(4):
            variants.append(np.rot90(base, k=k, axes=(1, 2)))
    return np.stack(variants)


def to_features(images: np.ndarray, metric: str = "l2") -> np.ndarray:
    """Flatten to (N, D), applying whatever normalisation the metric implies."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}, expected one of {METRICS}.")

    features = images.reshape(len(images), -1).astype(np.float32)

    if metric == "corr":
        features = features - features.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-8)

    return np.ascontiguousarray(features)


def _squared_distances(
    queries: np.ndarray, refs: np.ndarray, ref_sq: np.ndarray, chunk: int
) -> np.ndarray:
    query_sq = np.einsum("ij,ij->i", queries, queries)[:, None]
    out = np.empty((len(queries), len(refs)), dtype=np.float32)

    for start in range(0, len(refs), chunk):
        stop = min(start + chunk, len(refs))
        block = queries @ refs[start:stop].T
        out[:, start:stop] = query_sq + ref_sq[None, start:stop] - 2.0 * block

    return np.maximum(out, 0.0, out=out)


def nearest_neighbours(
    queries: np.ndarray,
    refs: np.ndarray,
    k: int = 5,
    metric: str = "l2",
    dihedral: bool = True,
    exclude_self: bool = False,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-`k` closest reference images for each query.

    `queries` and `refs` are float image arrays (N, H, W, 3) in [0, 1].
    Returns (distances, indices, variant) each of shape (n_queries, k), where
    `variant` records which of the 8 dihedral transforms of the query matched.
    """
    query_variants = dihedral_variants(queries) if dihedral else queries[None]
    ref_features = to_features(refs, metric)
    ref_sq = np.einsum("ij,ij->i", ref_features, ref_features)

    best = None
    best_variant = None

    for index, variant in enumerate(query_variants):
        distances = _squared_distances(
            to_features(variant, metric), ref_features, ref_sq, chunk
        )

        if best is None:
            best = distances
            best_variant = np.full(distances.shape, index, dtype=np.int8)
        else:
            closer = distances < best
            best = np.where(closer, distances, best)
            best_variant = np.where(closer, index, best_variant)

    if exclude_self:
        np.fill_diagonal(best, np.inf)

    partition = np.argpartition(best, kth=k - 1, axis=1)[:, :k]
    partition_distances = np.take_along_axis(best, partition, axis=1)
    order = np.argsort(partition_distances, axis=1)

    indices = np.take_along_axis(partition, order, axis=1)
    distances = np.sqrt(np.take_along_axis(partition_distances, order, axis=1))
    variants = np.take_along_axis(best_variant, indices, axis=1)

    return distances, indices, variants


def summarise(distances: np.ndarray) -> dict[str, float]:
    """Distribution of the top-1 distance, which is what memorisation moves."""
    top1 = distances[:, 0]
    return {
        "min": float(top1.min()),
        "p01": float(np.percentile(top1, 1)),
        "p05": float(np.percentile(top1, 5)),
        "median": float(np.median(top1)),
        "mean": float(top1.mean()),
        "max": float(top1.max()),
    }


def suspicious_fraction(
    generated: np.ndarray, baseline: np.ndarray, percentile: float = 1.0
) -> tuple[float, float]:
    """Share of generated samples closer than the baseline's `percentile` tail.

    Under the null hypothesis of no memorisation this lands near
    `percentile / 100`. Substantially more means some samples are closer to the
    training set than real galaxies typically are.
    """
    threshold = float(np.percentile(baseline[:, 0], percentile))
    fraction = float((generated[:, 0] < threshold).mean())
    return fraction, threshold


def plot_neighbour_grid(
    queries: np.ndarray,
    refs: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    path: str | Path,
    n_rows: int = 12,
    title: str = "Generated samples and their nearest training images",
) -> None:
    """One row per query: the sample, then its `k` closest training images."""
    import matplotlib.pyplot as plt

    n_rows = min(n_rows, len(queries))
    k = distances.shape[1]

    figure, axes = plt.subplots(
        n_rows, k + 1, figsize=(1.5 * (k + 1), 1.6 * n_rows), squeeze=False
    )

    for row in range(n_rows):
        axes[row][0].imshow(np.clip(queries[row], 0.0, 1.0))
        axes[row][0].set_ylabel(f"#{row}", fontsize=7)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        if row == 0:
            axes[row][0].set_title("generated", fontsize=8)

        for column in range(k):
            axis = axes[row][column + 1]
            axis.imshow(np.clip(refs[indices[row, column]], 0.0, 1.0))
            axis.set_xlabel(f"{distances[row, column]:.2f}", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"NN {column + 1}", fontsize=8)

    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def plot_distance_histogram(
    generated: np.ndarray,
    baseline: np.ndarray,
    path: str | Path,
    metric: str = "l2",
    threshold: float | None = None,
) -> None:
    """Generated vs real top-1 distances. Overlap means no memorisation."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "grid.color": INK,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
    })
    figure, axis = plt.subplots(figsize=(7, 3.6), layout="constrained")

    lowest = min(generated[:, 0].min(), baseline[:, 0].min())
    highest = max(generated[:, 0].max(), baseline[:, 0].max())
    bins = np.linspace(lowest, highest, 50)

    axis.hist(
        baseline[:, 0], bins=bins, density=True, color="#8a8a8a", alpha=0.55,
        label="validation images",
    )
    axis.hist(
        generated[:, 0], bins=bins, density=True, histtype="step", linewidth=1.8,
        color="darkorange", label="generated samples",
    )

    if threshold is not None:
        axis.axvline(threshold, color=INK, linestyle="--", linewidth=1.1)
        axis.annotate(
            "closest 1% of\nvalidation images", (threshold, 1), xycoords=("data", "axes fraction"),
            textcoords="offset points", xytext=(-5, -5), ha="right", va="top",
            fontsize=10, color=INK,
        )

    axis.set_xlabel(f"distance to the closest training image ({metric})")
    axis.set_ylabel("density")
    axis.grid(True)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, labelcolor=INK)

    figure.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(figure)
