"""Sampling throughput of a trained model over batch sizes.

One warm-up batch per size (compilation), then timed batches. Run once per
XLA_FLAGS setting to compare autotuning levels.

Usage:
    XLA_FLAGS=--xla_gpu_autotune_level=0 python -m scripts.sampling_speed --batch-sizes 8 16 32 64
"""

import argparse
import os
import time
from pathlib import Path

import jax

from src.config import GUIDANCE_SCALE, IMAGE_SIZE, TIMESTEPS, WEIGHTS_PATH
from src.generate import Generator
from src.logs import green


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    generator = Generator.from_weights(
        args.weights, image_size=IMAGE_SIZE, timesteps=TIMESTEPS
    )
    print(green(
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r} | {jax.devices()} | "
        f"{args.steps} DDIM steps, guidance {GUIDANCE_SCALE}"
    ))

    for batch_size in args.batch_sizes:
        def run(seed):
            return generator.generate(batch_size, seed=seed, steps=args.steps)

        try:
            start = time.perf_counter()
            run(0)
            warmup = time.perf_counter() - start
            start = time.perf_counter()
            for seed in range(1, args.repeats + 1):
                run(seed)
            per_image = (time.perf_counter() - start) / (args.repeats * batch_size)
        except Exception as e:
            print(
                f"batch {batch_size:>3}: failed ({type(e).__name__}: {str(e).splitlines()[0][:120]})"
            )
            continue
        print(green(
            f"batch {batch_size:>3}: {per_image:.3f} s/image, {1 / per_image:.2f} images/s "
            f"(first batch incl. compile {warmup:.0f}s) → 1,760 images ≈ {1760 * per_image / 60:.0f} min"
        ))


if __name__ == "__main__":
    main()
