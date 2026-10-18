"""Real-vs-real floor for the per-epoch quality scores (KID, FID, density, coverage).

The reference is the validation set exactly as training builds it. The other
side is the same number of train-split images, drawn per class to match the
reference's labels, as the generated samples are. The result is saved and
reused until the split, image size, N or seed changes.

Usage:
    python -m scripts.quality_floor
"""

import argparse
import json

import numpy as np

from src.config import (BATCH_SIZE, DATA_PATH, FID_SAMPLES, IMAGE_SIZE,
                        QUALITY_FLOOR_PATH, SPLIT_SEED, VAL_FRACTION)
from src.data_loader import make_data_splits
from src.metrics import (Reference, format_scores, inception_features,
                         load_images, load_inception, quality_reference,
                         quality_scores)


def class_matched(train_idx, train_labels, ref_labels, rng) -> np.ndarray:
    picks = [
        rng.choice(train_idx[train_labels == c], np.sum(ref_labels == c), replace=False)
        for c in np.unique(ref_labels)
    ]
    return np.sort(np.concatenate(picks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    splits = make_data_splits(
        DATA_PATH, BATCH_SIZE, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
        image_size=IMAGE_SIZE,
    )
    val_images, val_labels = quality_reference(splits, FID_SAMPLES)
    key = {
        "val_split_hash": splits.val_hash,
        "image_size": list(IMAGE_SIZE),
        "n": len(val_images),
        "seed": args.seed,
    }

    if QUALITY_FLOOR_PATH.exists() and not args.force:
        saved = json.loads(QUALITY_FLOOR_PATH.read_text())
        if {k: saved.get(k) for k in key} == key:
            print(f"Cached in {QUALITY_FLOOR_PATH} (--force to recompute):")
            print(f"  {format_scores(saved)}")
            return

    floor_idx = class_matched(
        splits.train_idx, splits.train_labels, val_labels,
        np.random.default_rng(args.seed),
    )

    inception = load_inception()
    ref = Reference(inception_features(inception, val_images))
    feats = inception_features(inception, load_images(DATA_PATH, floor_idx, IMAGE_SIZE))
    scores = quality_scores(feats, ref)

    QUALITY_FLOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_FLOOR_PATH.write_text(json.dumps({**key, **scores}, indent=2) + "\n")
    print(f"Real train vs val, N = {key['n']}: {format_scores(scores)}")
    print(f"Saved to {QUALITY_FLOOR_PATH}")


if __name__ == "__main__":
    main()
