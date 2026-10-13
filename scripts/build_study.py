"""Generate a blind pairwise study comparing two checkpoints.

Writes an image bank plus a blinded manifest. Generation happens here, once,
because sampling during rating would stall the UI and let waiting time bias the
answers. The manifest the browser reads never names a model; the answer key is
written separately and is not loaded by the page.

Usage:
    python -m scripts.build_study
    python -m scripts.build_study --pairs 10 --catch 2 -g 3.0
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    CLASS_NAMES,
    DATA_PATH,
    DDIM_STEPS,
    GUIDANCE_SCALE,
    IMAGE_SIZE,
    NUM_CLASSES,
    RUNS_DIR,
    STUDY_DIR,
    TIMESTEPS,
)
from src.generate import Generator
from src.memorization import load_dataset, load_labels

TEMPLATE = Path(__file__).resolve().parent.parent / "src" / "static" / "study.html"


def sample_for(path, labels, seed, batch, steps, guidance, name):
    print(f"  loading {name} from {path} ...")
    generator = Generator.from_weights(path, image_size=IMAGE_SIZE, timesteps=TIMESTEPS)
    if not generator.conditional:
        raise SystemExit(f"{path} is unconditional; both checkpoints must be conditional.")

    start = time.time()
    images = generator.generate_batched(
        len(labels), batch, seed, steps=steps, labels=labels, guidance=guidance
    )
    print(f"  {name}: {len(images)} images in {time.time() - start:.1f}s")

    del generator
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, default=RUNS_DIR / "old" / "galaxy_unet.safetensors")
    parser.add_argument("--b", type=Path, default=RUNS_DIR / "new" / "galaxy_unet.safetensors")
    parser.add_argument("--pairs", type=int, default=6, help="A-vs-B trials per class")
    parser.add_argument("--catch", type=int, default=2, help="real-vs-model trials per class per model")
    parser.add_argument("-g", "--guidance", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data", type=Path, default=Path(DATA_PATH))
    parser.add_argument("--out", type=Path, default=STUDY_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    ab_labels = np.repeat(np.arange(NUM_CLASSES), args.pairs)
    catch_labels = np.repeat(np.arange(NUM_CLASSES), args.catch)

    n_ab, n_catch = len(ab_labels), len(catch_labels)
    print(
        f"Study: {n_ab} A-vs-B + {n_catch} real-vs-A + {n_catch} real-vs-B "
        f"= {n_ab + 2 * n_catch} trials"
    )
    print(f"Generating {2 * n_ab + 2 * n_catch} images "
          f"(guidance {args.guidance}, {args.steps} steps)\n")

    a_ab = sample_for(args.a, ab_labels, args.seed, args.batch, args.steps, args.guidance, "A")
    a_catch = sample_for(args.a, catch_labels, args.seed + 10**6, args.batch, args.steps, args.guidance, "A (catch)")
    b_ab = sample_for(args.b, ab_labels, args.seed, args.batch, args.steps, args.guidance, "B")
    b_catch = sample_for(args.b, catch_labels, args.seed + 10**6, args.batch, args.steps, args.guidance, "B (catch)")

    print("\n  loading real images ...")
    cache = RUNS_DIR / "memorization" / f"refs_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_all.npy"
    dataset = load_dataset(args.data, IMAGE_SIZE, cache_path=cache)
    dataset_labels = load_labels(args.data)

    def real_for(class_id):
        pool = np.flatnonzero(dataset_labels == class_id)
        return (np.clip(dataset[rng.choice(pool)], 0.0, 1.0) * 255).astype(np.uint8)

    trials = []
    for i, class_id in enumerate(ab_labels):
        trials.append({"type": "ab", "class_id": int(class_id),
                       "A": a_ab[i], "B": b_ab[i]})
    for i, class_id in enumerate(catch_labels):
        trials.append({"type": "real_vs_a", "class_id": int(class_id),
                       "A": a_catch[i], "real": real_for(class_id)})
    for i, class_id in enumerate(catch_labels):
        trials.append({"type": "real_vs_b", "class_id": int(class_id),
                       "B": b_catch[i], "real": real_for(class_id)})

    rng.shuffle(trials)

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_trials, key_trials = [], {}
    for index, trial in enumerate(trials):
        trial_id = f"t{index:03d}"
        sides = [k for k in ("A", "B", "real") if k in trial]
        if rng.random() < 0.5:
            sides = sides[::-1]

        for position, side in zip(("l", "r"), sides):
            Image.fromarray(trial[side]).save(images_dir / f"{trial_id}_{position}.png")

        manifest_trials.append({
            "id": trial_id,
            "class_id": trial["class_id"],
            "class_name": CLASS_NAMES[trial["class_id"]],
            "left": f"images/{trial_id}_l.png",
            "right": f"images/{trial_id}_r.png",
        })
        key_trials[trial_id] = {
            "type": trial["type"],
            "class_id": trial["class_id"],
            "left": sides[0],
            "right": sides[1],
        }

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "created": created,
        "task": "fidelity",
        "n_trials": len(manifest_trials),
        "trials": manifest_trials,
    }
    key = {
        "created": created,
        "models": {"A": str(args.a), "B": str(args.b)},
        "guidance": args.guidance,
        "steps": args.steps,
        "seed": args.seed,
        "trials": key_trials,
    }

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "key.json").write_text(json.dumps(key, indent=2))
    shutil.copy(TEMPLATE, args.out / "index.html")

    print(f"\n  {args.out}/index.html   ({len(manifest_trials)} trials)")
    print(f"  {args.out}/manifest.json  (blinded)")
    print(f"  {args.out}/key.json       (answers - do not open)")
    print(f"\nRun the study:\n  cd {args.out} && python -m http.server 8080")


if __name__ == "__main__":
    main()
