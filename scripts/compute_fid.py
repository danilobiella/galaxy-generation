"""FID and KID of generated galaxies against the validation split.

Each score is printed next to a floor: the same number of real train-split
images scored against the same validation set.

Usage:
    python -m scripts.compute_fid -n 100
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.config import (DATA_PATH, DDIM_STEPS, GUIDANCE_SCALE, IMAGE_SIZE,
                        SPLIT_SEED, TIMESTEPS, VAL_FRACTION, WEIGHTS_PATH)
from src.data_loader import read_labels, stratified_split
from src.generate import Generator
from src.metrics import (format_scores, inception_features, load_images,
                         load_inception, quality_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--n-samples", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--steps", type=int, default=DDIM_STEPS)
    parser.add_argument("-g", "--guidance", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    args = parser.parse_args()
    n = args.n_samples
    rng = np.random.default_rng(args.seed)

    labels = read_labels(DATA_PATH)
    train_idx, val_idx = stratified_split(labels, VAL_FRACTION, SPLIT_SEED)
    floor_idx = np.sort(rng.choice(train_idx, n, replace=False))
    gen_labels = rng.choice(labels[val_idx], n)

    generator = Generator.from_weights(
        args.weights, image_size=IMAGE_SIZE, timesteps=TIMESTEPS
    )
    print(f"Generating {n} images ({args.steps} DDIM steps, guidance {args.guidance})...")
    start = time.time()
    generated = np.concatenate([
        generator.generate(
            n_samples=len(chunk),
            seed=args.seed + i,
            steps=args.steps,
            labels=chunk if generator.conditional else None,
            guidance=args.guidance,
        )
        for i, chunk in enumerate(np.array_split(gen_labels, -(-n // args.chunk)))
    ])
    print(f"Generated in {time.time() - start:.1f}s")

    inception = load_inception()
    val = inception_features(inception, load_images(DATA_PATH, val_idx, IMAGE_SIZE))
    floor = inception_features(inception, load_images(DATA_PATH, floor_idx, IMAGE_SIZE))
    gen = inception_features(inception, generated)

    print(f"\nAgainst {len(val)} validation images, N = {n}:")
    for name, feats in (("generated", gen), ("real (floor)", floor)):
        print(f"  {name:<13} {format_scores(quality_scores(feats, val, args.seed))}")


if __name__ == "__main__":
    main()
