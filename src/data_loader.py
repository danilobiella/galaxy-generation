import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import grain.python as grain
import lmdb
import numpy as np


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
        self._env = None  # Environment will be opened lazily in workers

    def __len__(self) -> int:
        return len(self.indices)

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.db_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._env

    def __getitem__(self, record_key: int) -> Dict[str, Any]:
        real_idx = self.indices[record_key]
        env = self._get_env()

        with env.begin() as txn:
            byteflow = txn.get(f"{real_idx}".encode())
            sample = pickle.loads(byteflow)

        return {"image": sample["img"], "label": sample["label"]}


def process(features):
    img = features["image"].astype(np.float32) / 255.0
    features["image"] = (img * 2.0) - 1.0
    features["label"] = np.int32(features["label"])
    return (features["image"], features["label"])


#
def make_data_loaders(
    db_path: str, batch_size: int, seed: int = 42
) -> Tuple[grain.DataLoader, grain.DataLoader]:
    # 1. Peek at the metadata once to get total count
    temp_env = lmdb.open(db_path, readonly=True, lock=False)
    with temp_env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        total_samples = meta["num_samples"]
    temp_env.close()

    indices = np.arange(total_samples)

    source = GalaxyLMDBDataSource(db_path, indices)
    return (
        grain.MapDataset.source(source)
        .shuffle(seed=seed)
        .map(process)
        .batch(batch_size=batch_size, drop_remainder=True)
    )


def make_overfit_loader(
    db_path: str,
    batch_size: int,
    target_class: int = 5,  # Defaulting to 5 (Barred Spirals) - pick your favorite!
    num_samples: int = 5,
) -> grain.MapDataset:
    """
    Creates a loader that infinitely repeats the SAME 5 images from the SAME class.
    Perfect for checking if the model has enough capacity to 'memorize' details.
    """
    # 1. Find indices belonging to the target class
    temp_env = lmdb.open(db_path, readonly=True, lock=False)
    found_indices = []

    print(f"Searching for {num_samples} samples of class {target_class}...")

    with temp_env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        total_available = meta["num_samples"]

        # Scan until we find our 5 targets
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

    # 2. Create the source with ONLY these indices
    overfit_indices = np.array(found_indices)
    source = GalaxyLMDBDataSource(db_path, overfit_indices)

    # 3. Build the infinite pipeline
    return (
        grain.MapDataset.source(source)
        .repeat()  # This makes the 5 images loop forever
        .map(process)
        .batch(batch_size=batch_size, drop_remainder=False)
    )
