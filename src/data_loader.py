import pickle
import threading
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import grain.python as grain
import lmdb
import numpy as np
from PIL import Image


@dataclass
class GalaxyLMDBDataSource(grain.RandomAccessDataSource):
    db_path: str
    indices: np.ndarray

    def __post_init__(self):
        env = lmdb.open(self.db_path, readonly=True, lock=False)
        with env.begin() as txn:
            meta_bytes = txn.get(b"__metadata__")
            if meta_bytes is None:
                raise ValueError("LMDB does not contain '__metadata__' key!")
            self.metadata = pickle.loads(meta_bytes)
        env.close()
        self._env = None
        self._env_lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.indices)

    def _get_env(self):
        if self._env is not None:
            return self._env

        with self._env_lock:
            if self._env is None:
                self._env = lmdb.open(
                    self.db_path,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state.pop("_env_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = None
        self._env_lock = threading.Lock()

    def __getitem__(self, record_key: int) -> Dict[str, Any]:
        real_idx = self.indices[record_key]
        env = self._get_env()

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


def make_data_loaders(
    db_path: str,
    batch_size: int,
    seed: int = 42,
    image_size: Optional[Tuple[int, int]] = None,
    augment: bool = True,
) -> TrainingData:
    temp_env = lmdb.open(db_path, readonly=True, lock=False)
    with temp_env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        total_samples = meta["num_samples"]
    temp_env.close()

    indices = np.arange(total_samples)
    source = GalaxyLMDBDataSource(db_path, indices)

    process_fn = partial(process_features, image_size=image_size)

    dataset = grain.MapDataset.source(source).seed(seed).shuffle().map(process_fn)

    if augment:
        dataset = dataset.random_map(augment_features)

    epoch = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return TrainingData(epoch.repeat(None), len(epoch))


def make_overfit_loader(
    db_path: str,
    batch_size: int,
    target_class: int = 5,
    num_samples: int = 5,
    image_size: Optional[Tuple[int, int]] = None,
) -> grain.MapDataset:
    temp_env = lmdb.open(db_path, readonly=True, lock=False)
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
    temp_env.close()

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
