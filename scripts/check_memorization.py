"""Check whether the model is copying its training data.

Generates samples and finds their nearest neighbours in the training split. The
baseline is the same search with the validation images as queries: real galaxies
the model never saw. If generated samples sit no closer to the training set than
those do, there is no evidence of memorisation. Writes a neighbour contact sheet
and a distance histogram.

Usage:
    python -m scripts.check_memorization
    python -m scripts.check_memorization -n 256 --metric corr --k 5
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.config import (BATCH_SIZE, DATA_PATH, DDIM_STEPS, IMAGE_SIZE,
                        RESULTS_DIR, SPLIT_SEED, TIMESTEPS, VAL_FRACTION,
                        WEIGHTS_PATH)
from src.data_loader import make_data_splits
from src.generate import Generator, random_seed
from src.memorization import (METRICS, nearest_neighbours,
                              plot_distance_histogram, plot_neighbour_grid,
                              summarise, suspicious_fraction, to_float)
from src.metrics import load_images, quality_reference


def format_stats(name: str, stats: dict[str, float]) -> str:
    body = "  ".join(f"{key}={value:.4f}" for key, value in stats.items())
    return f"  {name:<12} {body}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--n-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=DDIM_STEPS)
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument("--metric", choices=METRICS, default="l2")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--no-dihedral", dest="dihedral", action="store_false")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "memorization")
    args = parser.parse_args()

    seed = random_seed() if args.seed is None else args.seed
    args.out.mkdir(parents=True, exist_ok=True)

    splits = make_data_splits(
        DATA_PATH, BATCH_SIZE, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
        image_size=IMAGE_SIZE,
    )
    print(f"Loading {len(splits.train_idx)} training images ...")
    start = time.time()
    refs = to_float(load_images(DATA_PATH, splits.train_idx, IMAGE_SIZE))
    holdout = to_float(quality_reference(splits)[0])
    print(f"  done in {time.time() - start:.1f}s; {len(holdout)} validation images as baseline")

    generator = Generator.from_weights(
        args.weights, image_size=IMAGE_SIZE, timesteps=TIMESTEPS
    )
    print(f"Sampling {args.n_samples} images ({args.steps} DDIM steps) ...")
    start = time.time()
    generated = to_float(
        generator.generate_batched(
            args.n_samples, args.batch_size, seed=seed, steps=args.steps
        )
    )
    print(f"  done in {time.time() - start:.1f}s")

    search = dict(k=args.k, metric=args.metric, dihedral=args.dihedral, chunk=4096)

    print(f"\nSearching ({args.metric}, dihedral={args.dihedral}) ...")
    start = time.time()
    gen_distances, gen_indices, gen_variants = nearest_neighbours(
        generated, refs, **search
    )
    base_distances, _, _ = nearest_neighbours(holdout, refs, **search)
    print(f"  done in {time.time() - start:.1f}s")

    fraction, threshold = suspicious_fraction(gen_distances, base_distances)

    print("\nTop-1 distance to the training set")
    print(format_stats("generated", summarise(gen_distances)))
    print(format_stats("validation", summarise(base_distances)))
    print(
        f"\n  {fraction:.1%} of generated samples fall below the validation p01 "
        f"({threshold:.4f}); {0.01:.1%} expected if the model is not copying"
    )

    if args.dihedral:
        rotated = int((gen_variants[:, 0] != 0).sum())
        print(f"  {rotated}/{len(generated)} best matches needed a rotation or flip")

    order = np.argsort(gen_distances[:, 0])
    grid_path = args.out / f"neighbours_{args.metric}_{seed}.png"
    hist_path = args.out / f"distances_{args.metric}_{seed}.png"

    plot_neighbour_grid(
        generated[order],
        refs,
        gen_distances[order],
        gen_indices[order],
        grid_path,
        n_rows=args.rows,
        title=f"Closest {args.rows} of {args.n_samples} samples ({args.metric})",
    )
    plot_distance_histogram(
        gen_distances, base_distances, hist_path, args.metric, threshold
    )

    print(f"\n  {grid_path}")
    print(f"  {hist_path}")


if __name__ == "__main__":
    main()
