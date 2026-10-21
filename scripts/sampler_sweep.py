"""KID and FID of a trained model over a grid of guidance scales and DDIM steps.

Scores against the same validation reference, labels, noise and KID seed as the
per-epoch training scores, so guidance 2.0 at 20 steps on the final EMA weights
reproduces the run's last KID/FID, and every row compares to the baseline from
`scripts.quality_baseline`. With `-n` below the full reference, only KID stays
comparable to the baseline.

Usage:
    python -m scripts.sampler_sweep --weights runs/good_run/galaxy_unet_ema.safetensors
"""

import argparse
import json
from pathlib import Path

from src.config import (BATCH_SIZE, DATA_PATH, FID_BATCH_SIZE, FID_SAMPLES,
                        IMAGE_SIZE, RESULTS_DIR, SPLIT_SEED, TIMESTEPS,
                        VAL_FRACTION, WEIGHTS_PATH)
from src.data_loader import make_data_splits
from src.generate import Generator
from src.logs import green
from src.metrics import (Reference, eval_keys, format_scores, format_timings,
                         inception_features, load_baseline, load_inception,
                         quality_reference, score_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("-g", "--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--steps", type=int, nargs="+", default=[20, 50])
    parser.add_argument("-n", "--n-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=FID_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "sampler_sweep.json")
    args = parser.parse_args()

    splits = make_data_splits(
        DATA_PATH, BATCH_SIZE, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
        image_size=IMAGE_SIZE,
    )
    val_images, val_labels = quality_reference(splits, FID_SAMPLES)
    n = args.n_samples or len(val_images)
    _, quality_key = eval_keys(SPLIT_SEED)

    generator = Generator.from_weights(args.weights, image_size=IMAGE_SIZE, timesteps=TIMESTEPS)
    labels = val_labels[:n] if generator.conditional else None
    inception = load_inception()
    ref = Reference(inception_features(inception, val_images, args.batch_size))

    baseline = load_baseline(splits.val_hash)
    if baseline is not None:
        print(f"Baseline (real vs real, N = {baseline['n']}): {format_scores(baseline)}")
    print(f"{args.weights}, N = {n} against {len(val_images)} validation images\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for steps in args.steps:
        for guidance in args.guidance:
            scores, timings = score_samples(
                generator.samplers, generator.params, inception, ref, quality_key,
                n, labels, steps, guidance, args.batch_size,
            )
            rows.append({"guidance": guidance, "steps": steps, **scores, "seconds": timings})
            print(green(
                f"guidance {guidance:<4} steps {steps:<4} {format_scores(scores)} "
                f"({format_timings(timings)})"
            ))
            args.out.write_text(json.dumps({
                "weights": str(args.weights), "val_split_hash": splits.val_hash,
                "n": n, "rows": rows,
            }, indent=2) + "\n")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
