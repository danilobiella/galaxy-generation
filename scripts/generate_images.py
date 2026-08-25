"""Generate galaxy images from the trained model.

Usage:
    python -m scripts.generate_images                  # one image, random seed
    python -m scripts.generate_images -n 10 --seed 42
"""

import argparse
import time
from pathlib import Path

from PIL import Image

from config import (
    CLASS_NAMES,
    DDIM_ETA,
    DDIM_STEPS,
    GUIDANCE_SCALE,
    IMAGE_SIZE,
    RUNS_DIR,
    SAMPLER,
    TIMESTEPS,
    WEIGHTS_PATH,
)
from generate import SAMPLERS, Generator, random_seed

DEFAULT_OUT_DIR = RUNS_DIR / "generated"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample images from the diffusion model."
    )
    parser.add_argument("-n", "--n-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sampler", choices=SAMPLERS, default=SAMPLER)
    parser.add_argument("--steps", type=int, default=DDIM_STEPS)
    parser.add_argument("--eta", type=float, default=DDIM_ETA)
    parser.add_argument("--no-clip", dest="clip", action="store_false")
    parser.add_argument("-c", "--class-id", type=int, default=None)
    parser.add_argument("-g", "--guidance", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    seed = random_seed() if args.seed is None else args.seed
    n_steps = TIMESTEPS if args.sampler == "ddpm" else args.steps

    print(f"Loading {args.weights} ...")
    generator = Generator.from_weights(
        args.weights, image_size=IMAGE_SIZE, timesteps=TIMESTEPS
    )

    if generator.conditional:
        label = "cycling classes" if args.class_id is None else (
            f"class {args.class_id} ({CLASS_NAMES[args.class_id]})"
        )
        print(f"Conditional checkpoint: {label}, guidance {args.guidance}")
    elif args.class_id is not None:
        parser_error = "Checkpoint is unconditional; --class-id not supported."
        raise SystemExit(parser_error)

    print(
        f"Sampling {args.n_samples} image(s) with {args.sampler} "
        f"over {n_steps} steps (seed {seed}) ..."
    )
    start = time.time()
    images = generator.generate(
        n_samples=args.n_samples,
        seed=seed,
        sampler=args.sampler,
        steps=args.steps,
        eta=args.eta,
        clip=args.clip,
        labels=args.class_id,
        guidance=args.guidance,
    )
    print(f"Done in {time.time() - start:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        path = args.out / f"galaxy_{args.sampler}{n_steps}_{seed}_{i:02d}.png"
        Image.fromarray(image).save(path)
        print(f"  {path}")


if __name__ == "__main__":
    main()
