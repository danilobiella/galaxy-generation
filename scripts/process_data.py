"""Convert the Galaxy10 DECals HDF5 release into an LMDB keyed by sample index."""

import pickle

import h5py
import lmdb
import numpy as np
from tqdm import tqdm

from src.config import LOCAL_DATA_PATH

H5_PATH = "data/Galaxy10_DECals.h5"
METADATA_KEY = b"__metadata__"


def load_h5(h5_path: str) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading {h5_path} into memory...")
    with h5py.File(h5_path, "r") as h5:
        images, labels = h5["images"][:], h5["ans"][:]

    if len(images) != len(labels):
        raise ValueError(
            f"{h5_path} has {len(images)} images but {len(labels)} labels"
        )

    return images, labels


def write_samples(
    txn: lmdb.Transaction, images: np.ndarray, labels: np.ndarray
) -> None:
    for i in tqdm(range(len(images)), desc="Writing to LMDB", unit="img"):
        sample = {"img": images[i], "label": int(labels[i])}
        txn.put(f"{i}".encode(), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))


def convert_h5_to_lmdb(h5_path: str, lmdb_path: str) -> None:
    images, labels = load_h5(h5_path)

    print(f"Starting conversion of {len(images)} samples...")
    env = lmdb.open(
        lmdb_path, map_size=10 * 1024**3, writemap=True, map_async=True, meminit=False
    )
    with env, env.begin(write=True) as txn:
        write_samples(txn, images, labels)
        metadata = {
            "num_samples": len(images),
            "shape": images.shape[1:],
            "dtype": str(images.dtype),
        }
        txn.put(METADATA_KEY, pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))

    print("\nConversion complete!")


if __name__ == "__main__":
    convert_h5_to_lmdb(H5_PATH, LOCAL_DATA_PATH)
