"""Grid of generated samples: classes, several samples per class, and guidance
scales.

By default rows are classes (each repeated --per-class times with different
noise) and columns are guidance scales. Every cell in a row shares the same
initial noise, so differences along a row come from guidance alone.
--transpose puts the classes on columns and the samples (times scales) on rows,
drawn as one gap-separated mosaic.
--mirror LEFT RIGHT puts w = 0 in the middle and increases w outwards, towards
class LEFT on the left and class RIGHT on the right. At w = 0 the label is
ignored, so both halves share the middle column.

Usage:
    python -m scripts.sample_grid                                        # classes x guidance
    python -m scripts.sample_grid --classes 8 --per-class 16 --scales 0 1 4 16
    python -m scripts.sample_grid --scales 1 --per-class 6 --steps 50 --transpose -o assets/class_showcase.png
    python -m scripts.sample_grid --mirror 8 1 --per-class 8 --scales 0 1 2 4 8 --steps 50
"""

import argparse
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import (CLASS_NAMES, FID_STEPS, IMAGE_SIZE, NUM_CLASSES,
                        RESULTS_DIR, TIMESTEPS, WEIGHTS_PATH)
from src.generate import Generator, random_seed
from src.logs import format_duration

DEFAULT_SCALES = (0.0, 1.0, 1.5, 2.0, 3.0, 5.0)
UPSCALE = 2
GAP = 4
LABEL_WIDTH = 13
INK = "#8b949e"
SAVE_SCALE = 2


def mosaic(cells: np.ndarray) -> np.ndarray:
    """(rows, cols, H, W, 3) uint8 cells as one RGBA image with transparent
    gaps, each cell upscaled by repetition so it stays sharp when the page
    shrinks it."""
    cells = cells.repeat(UPSCALE, axis=2).repeat(UPSCALE, axis=3)
    n_rows, n_cols, height, width, _ = cells.shape
    image = np.zeros(
        (n_rows * (height + GAP) - GAP, n_cols * (width + GAP) - GAP, 4), np.uint8
    )
    for row in range(n_rows):
        for col in range(n_cols):
            top, left = row * (height + GAP), col * (width + GAP)
            image[top:top + height, left:left + width, :3] = cells[row, col]
            image[top:top + height, left:left + width, 3] = 255
    return image


def plot_transposed(grid, classes, per_class, scales, path) -> None:
    n_classes, n_scales = len(classes), len(scales)
    cells = grid.reshape(n_classes, per_class, n_scales, *grid.shape[2:])
    cells = cells.transpose(1, 2, 0, 3, 4, 5).reshape(per_class * n_scales, n_classes, *grid.shape[2:])
    image = mosaic(cells)

    dpi = 100
    header = 58
    margin = 70 if n_scales > 1 else 0
    figure = plt.figure(figsize=((image.shape[1] + margin) / dpi, (image.shape[0] + header) / dpi), dpi=dpi)
    axis = figure.add_axes((
        margin / (image.shape[1] + margin), 0,
        image.shape[1] / (image.shape[1] + margin), image.shape[0] / (image.shape[0] + header),
    ))
    axis.imshow(image, interpolation="nearest")
    axis.set_axis_off()

    cell = IMAGE_SIZE[1] * UPSCALE + GAP
    for col, class_id in enumerate(classes):
        axis.text(
            col * cell + (cell - GAP) / 2, -10, textwrap.fill(CLASS_NAMES[class_id], LABEL_WIDTH),
            ha="center", va="bottom", fontsize=12, color=INK,
        )
    if n_scales > 1:
        for row in range(per_class * n_scales):
            axis.text(
                -10, row * cell + (cell - GAP) / 2, f"w = {scales[row % n_scales]:g}",
                ha="right", va="center", fontsize=12, color=INK,
            )

    figure.savefig(path, dpi=dpi * SAVE_SCALE, transparent=True)
    plt.close(figure)


def plot_mirrored(left_grid, right_grid, left, right, scales, path) -> None:
    cells = np.concatenate([left_grid[:, :0:-1], right_grid], axis=1)
    labels = [f"{w:g}" for w in scales[:0:-1]] + [f"w = {scales[0]:g}"] + [f"{w:g}" for w in scales[1:]]
    image = mosaic(cells)

    dpi = 100
    header = 70
    figure = plt.figure(figsize=(image.shape[1] / dpi, (image.shape[0] + header) / dpi), dpi=dpi)
    axis = figure.add_axes((0, 0, 1, image.shape[0] / (image.shape[0] + header)))
    axis.imshow(image, interpolation="nearest")
    axis.set_axis_off()

    cell = IMAGE_SIZE[1] * UPSCALE + GAP
    for col, label in enumerate(labels):
        axis.text(col * cell + (cell - GAP) / 2, -8, label, ha="center", va="bottom", fontsize=13, color=INK)
    half = len(scales) - 1
    axis.text(half * cell / 2, -36, f"← {CLASS_NAMES[left]}", ha="center", va="bottom", fontsize=14, color=INK)
    axis.text((half + 1 + half / 2) * cell, -36, f"{CLASS_NAMES[right]} →", ha="center", va="bottom",
              fontsize=14, color=INK)

    figure.savefig(path, dpi=dpi * SAVE_SCALE, transparent=True)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=FID_STEPS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--mirror", type=int, nargs=2, metavar=("LEFT", "RIGHT"), default=None)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()

    seed = random_seed() if args.seed is None else args.seed
    classes = list(range(NUM_CLASSES)) if args.classes is None else args.classes
    if args.mirror is not None:
        classes = list(args.mirror)
        if sorted(args.scales) != args.scales or args.scales[0] != 0:
            raise SystemExit("--mirror needs ascending --scales starting at 0")
    rows = [c for c in classes for _ in range(args.per_class)]

    print(f"Loading {args.weights} ...")
    generator = Generator.from_weights(
        args.weights, image_size=IMAGE_SIZE, timesteps=TIMESTEPS
    )

    if not generator.conditional:
        raise SystemExit(
            "This checkpoint is unconditional. Train with NUM_CLASSES set first."
        )

    n_rows, n_cols = len(rows), len(args.scales)
    print(f"Sampling {n_rows}x{n_cols} images (seed {seed}, {args.steps} steps) ...")

    start = time.perf_counter()
    grid = np.empty((n_rows, n_cols, *IMAGE_SIZE, 3), dtype=np.uint8)
    block = args.per_class if args.mirror is not None else n_rows
    for column, scale in enumerate(args.scales):
        for first in range(0, n_rows, block):
            grid[first:first + block, column] = generator.generate_batched(
                n_samples=block,
                batch_size=args.batch_size,
                seed=seed,
                steps=args.steps,
                labels=rows[first:first + block],
                guidance=scale,
            )
    print(f"  done in {format_duration(time.perf_counter() - start)}")

    path = RESULTS_DIR / f"sample_grid_{seed}.png" if args.out is None else args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.mirror is not None:
        left_grid, right_grid = grid[:args.per_class], grid[args.per_class:]
        diff = np.abs(left_grid[:, 0].astype(int) - right_grid[:, 0].astype(int)).max()
        print(f"  w = 0 columns differ by at most {diff} grey levels")
        plot_mirrored(left_grid, right_grid, *classes, args.scales, path)
        print(f"\n  {path}")
        return
    if args.transpose:
        plot_transposed(grid, classes, args.per_class, args.scales, path)
        print(f"\n  {path}")
        return

    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.7 * n_cols, 1.7 * n_rows), squeeze=False
    )
    for row, class_id in enumerate(rows):
        for column, scale in enumerate(args.scales):
            axis = axes[row][column]
            axis.imshow(grid[row, column])
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"w = {scale:g}", fontsize=14, color=INK)
            if len(classes) > 1 and column == 0 and row % args.per_class == 0:
                axis.set_ylabel(CLASS_NAMES[class_id], fontsize=7, rotation=0, ha="right", va="center", color=INK)

    title = CLASS_NAMES[classes[0]] if len(classes) == 1 else "Classifier-free guidance sweep"
    figure.suptitle(title, fontsize=16, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    figure.savefig(path, dpi=130, bbox_inches="tight", transparent=True)
    plt.close(figure)
    print(f"\n  {path}")


if __name__ == "__main__":
    main()
