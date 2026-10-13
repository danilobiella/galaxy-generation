"""Training checkpoints, so an interrupted run can be resumed.

`TrainState.step` is the only thing that records progress. The RNG key, the
position in the data stream and the epoch number are all derived from it, so a
checkpoint holds the state plus a little JSON and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import orbax.checkpoint as ocp  # type: ignore


class Resumed(NamedTuple):
    train_state: Any
    history: dict
    meta: dict[str, Any]


def run_meta(
    *,
    steps_per_epoch: int,
    n_epochs: int,
    learning_rate: float,
    warmup_steps: int,
    final_learning_rate: float,
    data_seed: int,
    val_fraction: float,
    split_seed: int,
    val_split_hash: str,
) -> dict[str, Any]:
    """What a resume cannot change, because none of it is in the checkpoint.

    The learning-rate schedule is rebuilt from config and replayed by step, so
    every input to it has to match. `steps_per_epoch` counts twice over -- it
    sets `decay_steps` with `n_epochs`, and the epoch number is derived from it.
    `data_seed` is here because the stream is resumed by position: the same
    index into a differently shuffled epoch is a different sample. The split
    fields keep validation images from moving into training on a resume.
    """
    return {
        "steps_per_epoch": steps_per_epoch,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "final_learning_rate": final_learning_rate,
        "data_seed": data_seed,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "val_split_hash": val_split_hash,
    }


def manager(
    directory: str | Path, keep: int = 2, every: int = 1
) -> ocp.CheckpointManager:
    """A manager keyed by epoch. `every` is in epochs, and orbax itself decides
    whether a given `save` call writes, so the training loop has no interval
    arithmetic in it."""
    return ocp.CheckpointManager(
        Path(directory).resolve(),
        options=ocp.CheckpointManagerOptions(
            create=True, max_to_keep=keep, save_interval_steps=every
        ),
    )


def save(
    mgr: ocp.CheckpointManager,
    epoch: int,
    train_state: Any,
    history: dict,
    meta: dict[str, Any],
    force: bool = False,
) -> bool:
    return mgr.save(
        epoch,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(train_state),
            history=ocp.args.JsonSave(history),
            meta=ocp.args.JsonSave(meta),
        ),
        force=force,
    )


def saver(mgr: ocp.CheckpointManager, meta: dict[str, Any]) -> Callable:
    """The per-epoch callback `fit_model` wants, with the run's constants bound."""

    def save_checkpoint(epoch, train_state, history, is_last):
        save(mgr, epoch, train_state, history, meta, force=is_last)

    return save_checkpoint


def restore(mgr: ocp.CheckpointManager, abstract_state: Any) -> Resumed | None:
    """Restore the newest checkpoint, or None if there is none.

    `abstract_state` is a freshly initialised `TrainState`, used only for its
    shapes: restoring without a target returns optax's state as plain dicts
    rather than its own NamedTuples, and `optimizer.update` then fails on it.
    Only the structure is needed, so the target is reduced to shapes and dtypes
    and the restored arrays are not allocated twice.
    """
    epoch = mgr.latest_step()
    if epoch is None:
        return None

    target = jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=getattr(leaf, "sharding", None)
        ),
        abstract_state,
    )
    restored = mgr.restore(
        epoch,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(target),
            history=ocp.args.JsonRestore(),
            meta=ocp.args.JsonRestore(),
        ),
    )
    return Resumed(restored["state"], restored["history"], restored["meta"])


def check_resumable(saved: dict[str, Any], current: dict[str, Any]) -> None:
    changed = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    if changed:
        detail = ", ".join(
            f"{k}: checkpoint {was!r} vs now {now!r}"
            for k, (was, now) in changed.items()
        )
        raise ValueError(
            f"Cannot resume: the run no longer matches the checkpoint ({detail}). "
            "The learning-rate schedule is rebuilt from config and replayed from "
            "the step count, so these must agree or training would continue on a "
            "different curve."
        )
