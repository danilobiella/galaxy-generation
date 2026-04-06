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
    return features


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
