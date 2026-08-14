"""Generate galaxy images from the trained model.

Usage:
    python -m scripts.generate_images                  # one image, random seed
    python -m scripts.generate_images -n 10 --seed 42
"""

import argparse
from pathlib import Path

from PIL import Image

from config import IMAGE_SIZE, OUTPUT_DIR, TIMESTEPS, WEIGHTS_PATH
from generate import Generator, random_seed

DEFAULT_OUT_DIR = OUTPUT_DIR / "generated"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample images from the diffusion model."
    )
    parser.add_argument("-n", "--n-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    seed = random_seed() if args.seed is None else args.seed

    print(f"Loading {args.weights} ...")
    generator = Generator.from_weights(
        args.weights, image_size=IMAGE_SIZE, timesteps=TIMESTEPS
    )

    print(
        f"Sampling {args.n_samples} image(s) over {TIMESTEPS} steps (seed {seed}) ..."
    )
    images = generator.generate(n_samples=args.n_samples, seed=seed)

    args.out.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        path = args.out / f"galaxy_{seed}_{i:02d}.png"
        Image.fromarray(image).save(path)
        print(f"  {path}")


if __name__ == "__main__":
    main()
