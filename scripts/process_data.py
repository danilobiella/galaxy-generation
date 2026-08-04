import pickle

import h5py
import lmdb
from tqdm import tqdm


def convert_h5_to_lmdb(h5_path, lmdb_path):
    # 1. READ ENTIRE DATASET INTO RAM (This is the #1 speedup)
    print(f"Loading {h5_path} into memory...")
    with h5py.File(h5_path, "r") as h5:
        # Adding [:] at the end forces h5py to load the data into a Numpy array
        images = h5["images"][:]
        labels = h5["ans"][:]
        num_samples = images.shape[0]

    # 2. OPEN LMDB WITH OPTIMIZED FLAGS
    # writemap: Uses a writeable memory map (much faster)
    # map_async: Non-blocking disk flushes
    map_size = 10 * 1024**3
    env = lmdb.open(
        lmdb_path, map_size=map_size, writemap=True, map_async=True, meminit=False
    )

    print(f"Starting conversion of {num_samples} samples...")

    with env.begin(write=True) as txn:
        # Use pickle.HIGHEST_PROTOCOL for speed
        protocol = pickle.HIGHEST_PROTOCOL

        for i in tqdm(range(num_samples), desc="Writing to LMDB", unit="img"):
            sample = {"img": images[i], "label": int(labels[i])}

            # Using f-strings and standard encoding
            txn.put(f"{i}".encode(), pickle.dumps(sample, protocol=protocol))

        # Store metadata dynamically
        metadata = {
            "num_samples": num_samples,
            "shape": images.shape[1:],
            "dtype": str(images.dtype),
        }
        txn.put(b"__metadata__", pickle.dumps(metadata, protocol=protocol))

    print("\nConversion complete! Closing LMDB.")
    env.close()


if __name__ == "__main__":
    h5_path = "data/Galaxy10_DECals.h5"
    lmdb_path = "data/galaxy10_decals_lmdb"
    convert_h5_to_lmdb(h5_path, lmdb_path)
