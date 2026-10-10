import hashlib
import os
import pickle
import threading
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import grain.python as grain
import lmdb
import numpy as np
from PIL import Image


_ENVS: Dict[str, Tuple[int, lmdb.Environment]] = {}
_ENVS_LOCK = threading.Lock()


def open_env(db_path: str) -> lmdb.Environment:
    """One shared read-only handle per path, owned by the process that opened it.

    LMDB refuses to open the same environment twice in one process, so the
    train and validation sources must share it. Worker processes must be
    spawned, not forked: a forked child inherits LMDB's record of the open
    environment and can neither reopen it nor safely use the parent's handle.
    """
    key = os.path.realpath(db_path)
    with _ENVS_LOCK:
        if key not in _ENVS:
            env = lmdb.open(
                db_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            _ENVS[key] = (os.getpid(), env)
        owner, env = _ENVS[key]
    if owner != os.getpid():
        raise RuntimeError(
            f"LMDB handle for {key} was opened in process {owner} and inherited "
            "by fork. Start data-loading workers with the 'spawn' method."
        )
    return env


def close_envs() -> None:
    with _ENVS_LOCK:
        for owner, env in _ENVS.values():
            if owner == os.getpid():
                env.close()
        _ENVS.clear()


@dataclass
class GalaxyLMDBDataSource(grain.RandomAccessDataSource):
    db_path: str
    indices: np.ndarray

    def __post_init__(self):
        with open_env(self.db_path).begin() as txn:
            meta_bytes = txn.get(b"__metadata__")
            if meta_bytes is None:
                raise ValueError("LMDB does not contain '__metadata__' key!")
            self.metadata = pickle.loads(meta_bytes)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, record_key: int) -> Dict[str, Any]:
        real_idx = self.indices[record_key]
        env = open_env(self.db_path)

        with env.begin() as txn:
            byteflow = txn.get(f"{real_idx}".encode())
            sample = pickle.loads(byteflow)

        return {"image": sample["img"], "label": sample["label"]}


def process_features(
    features: Dict[str, Any], image_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure function for processing and resizing."""
    img = features["image"]

    if image_size is not None:
        target_h, target_w = image_size
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img = np.array(img_pil)

    img = img.astype(np.float32) / 255.0
    features["image"] = (img * 2.0) - 1.0
    features["label"] = np.int32(features["label"])

    return (features["image"], features["label"])


def dihedral_augment(
    image: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One of the 8 rotations/reflections of the image, chosen uniformly.

    A galaxy's orientation on the sky is arbitrary, so every element of the
    dihedral group of the square maps a training image to an equally valid one.
    """
    if bool(rng.integers(2)):
        image = image[:, ::-1]

    k = int(rng.integers(4))
    if k:
        image = np.rot90(image, k=k, axes=(0, 1))

    return np.ascontiguousarray(image)


def augment_features(
    features: Tuple[np.ndarray, np.ndarray], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    image, label = features
    return (dihedral_augment(image, rng), label)


@dataclass(frozen=True)
class TrainingData:
    """An endless stream of batches, plus how many of them make one epoch.

    `repeat` reseeds the upstream `shuffle` and `random_map` per epoch, so each
    pass is ordered and augmented differently. Batching before repeating keeps
    an epoch an exact pass over the data: the same number of trailing samples is
    dropped each epoch, but a different set of them, since the order changes.

    Repeating makes the dataset infinite, so `len()` no longer gives the epoch
    length and the count travels alongside the stream.
    """

    batches: grain.MapDataset
    steps_per_epoch: int


@dataclass(frozen=True)
class DataSplits:
    """A stratified train/validation split of one LMDB.

    `val` is finite, unshuffled and unaugmented, so every iteration yields the
    same batches. `val_hash` fingerprints the validation indices.
    """

    train: TrainingData
    val: grain.MapDataset
    val_hash: str


def _num_samples(db_path: str) -> int:
    with open_env(db_path).begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
    return meta["num_samples"]


def read_labels(db_path: str) -> np.ndarray:
    n = _num_samples(db_path)
    labels = np.empty(n, dtype=np.int64)
    with open_env(db_path).begin() as txn:
        for i in range(n):
            labels[i] = int(pickle.loads(txn.get(f"{i}".encode()))["label"])
    return labels


def stratified_split(
    labels: np.ndarray, val_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = int(np.floor(len(idx) * val_fraction + 0.5))
        if not 0 < n_val < len(idx):
            raise ValueError(
                f"Class {c} has {len(idx)} samples, which at "
                f"val_fraction={val_fraction} leaves {n_val} for validation and "
                f"{len(idx) - n_val} for training; both need at least one."
            )
        val.append(idx[:n_val])
        train.append(idx[n_val:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def split_hash(val_indices: np.ndarray) -> str:
    data = np.sort(np.asarray(val_indices)).astype(np.int64).tobytes()
    return hashlib.sha256(data).hexdigest()[:12]


def _training_data(
    source: GalaxyLMDBDataSource,
    batch_size: int,
    seed: int,
    image_size: Optional[Tuple[int, int]],
    augment: bool,
) -> TrainingData:
    process_fn = partial(process_features, image_size=image_size)

    dataset = grain.MapDataset.source(source).seed(seed).shuffle().map(process_fn)

    if augment:
        dataset = dataset.random_map(augment_features)

    epoch = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return TrainingData(epoch.repeat(None), len(epoch))


def make_data_loaders(
    db_path: str,
    batch_size: int,
    seed: int = 42,
    image_size: Optional[Tuple[int, int]] = None,
    augment: bool = True,
) -> TrainingData:
    indices = np.arange(_num_samples(db_path))
    source = GalaxyLMDBDataSource(db_path, indices)
    return _training_data(source, batch_size, seed, image_size, augment)


def make_data_splits(
    db_path: str,
    batch_size: int,
    val_fraction: float = 0.1,
    split_seed: int = 0,
    seed: int = 42,
    image_size: Optional[Tuple[int, int]] = None,
    augment: bool = True,
) -> DataSplits:
    train_idx, val_idx = stratified_split(
        read_labels(db_path), val_fraction, split_seed
    )

    train = _training_data(
        GalaxyLMDBDataSource(db_path, train_idx), batch_size, seed, image_size, augment
    )

    process_fn = partial(process_features, image_size=image_size)
    val = (
        grain.MapDataset.source(GalaxyLMDBDataSource(db_path, val_idx))
        .map(process_fn)
        .batch(batch_size=batch_size, drop_remainder=True)
    )
    if len(val) == 0:
        raise ValueError(
            f"Validation split has {len(val_idx)} samples, fewer than one batch "
            f"of {batch_size}."
        )

    return DataSplits(train, val, split_hash(val_idx))


def make_overfit_loader(
    db_path: str,
    batch_size: int,
    target_class: int = 5,
    num_samples: int = 5,
    image_size: Optional[Tuple[int, int]] = None,
) -> grain.MapDataset:
    temp_env = open_env(db_path)
    found_indices = []

    print(f"Searching for {num_samples} samples of class {target_class}...")

    with temp_env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        total_available = meta["num_samples"]

        for i in range(total_available):
            byteflow = txn.get(f"{i}".encode())
            sample = pickle.loads(byteflow)
            if int(sample["label"]) == target_class:
                found_indices.append(i)

            if len(found_indices) == num_samples:
                break

    if len(found_indices) < num_samples:
        raise ValueError(
            f"Could only find {len(found_indices)} samples for class {target_class}"
        )

    print(f"Sanity check indices found: {found_indices}")

    overfit_indices = np.array(found_indices)
    source = GalaxyLMDBDataSource(db_path, overfit_indices)

    process_fn = partial(process_features, image_size=image_size)

    return (
        grain.MapDataset.source(source)
        .repeat()
        .map(process_fn)
        .batch(batch_size=batch_size, drop_remainder=False)
    )
