from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore

from src.logs import format_duration, peak_memory

if TYPE_CHECKING:
    from src.diffusion import DiffusionSchedule, Samplers


class ModelState(NamedTuple):
    params: hk.Params


class TrainState(NamedTuple):
    model_state: ModelState
    opt_state: optax.OptState
    step: jnp.ndarray
    ema_params: hk.Params


@jax.jit
def get_global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))


def make_model_fns(
    unet: hk.Transformed,
    samplers: Samplers,
    image_shape: tuple[int, int, int] = (256, 256, 3),
    sample_steps: int = 20,
    num_classes: int | None = None,
    guidance: float = 1.0,
):
    def init_model_state(rng: jax.Array) -> ModelState:
        dummy_image = jnp.zeros((1, *image_shape))
        dummy_t = jnp.zeros((1,), dtype=jnp.int32)
        dummy_y = None if num_classes is None else jnp.zeros((1,), dtype=jnp.int32)

        dummy_input = (dummy_image, dummy_t, dummy_y)
        params = unet.init(rng, dummy_input)
        return ModelState(params=params)

    def generate_and_save_images(
        key: jax.Array,
        model_state: ModelState,
        filepath: str = "generated_images.png",
    ):
        n_rows, n_cols = 4, 4
        n_samples = n_rows * n_cols

        # One class per row, so the epoch figure shows conditioning working
        labels = None
        if num_classes is not None:
            labels = jnp.repeat(jnp.arange(n_rows) % num_classes, n_cols)

        latents = samplers.run_ddim(
            n_samples,
            sample_steps,
            True,
            model_state.params,
            key,
            0.0,
            labels,
            guidance,
        )

        images_np = jax.device_get(jnp.clip((latents + 1.0) / 2.0, 0.0, 1.0))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
        for idx in range(n_samples):
            row_idx, col_idx = divmod(idx, n_cols)
            ax = axes[row_idx, col_idx]
            ax.imshow(images_np[idx])
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated samples saved to: {filepath}")

    return init_model_state, generate_and_save_images


def noise_prediction_error(unet, schedule, params, x_0, t, epsilon, y):
    # Fetch cumulative alpha and reshape for broadcasting (B, 1, 1, 1)
    a_hat_t = schedule.alpha_hat[t][:, None, None, None]

    # Forward process: Corrupt images to timestep t (Reparameterization Trick)
    x_t = jnp.sqrt(a_hat_t) * x_0 + jnp.sqrt(1.0 - a_hat_t) * epsilon

    # Predict the added noise using the U-Net
    eps_hat = unet.apply(params, None, (x_t, t, y))

    return jnp.square(epsilon - eps_hat)


def make_train_fns(
    unet: hk.Transformed,
    optimizer: optax.GradientTransformation,
    schedule: DiffusionSchedule,
    num_classes: int | None = None,
    p_uncond: float = 0.1,
    ema_decay: float = 0.9999,
):
    """Build the training step.

    `ema_decay` maintains a second copy of the weights, an exponential moving
    average of the trained ones. Sampling from the average rather than the last
    iterate is standard for diffusion models and usually helps more than any
    learning-rate change. The average is never optimised, only tracked.
    """

    def ema_update(ema, params, step):
        warmup = (1.0 + step) / (10.0 + step)
        decay = jnp.minimum(ema_decay, warmup)
        return jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p, ema, params
        )

    def init_train_state(model_state: ModelState) -> TrainState:
        opt_state = optimizer.init(model_state.params)
        return TrainState(
            model_state=model_state,
            opt_state=opt_state,
            step=jnp.array(0, dtype=jnp.int32),
            ema_params=jax.tree_util.tree_map(jnp.copy, model_state.params),
        )

    @jax.jit
    def train_step(key, state, real_imgs, labels=None):
        def loss_fn(params, key):
            t_key, noise_key, drop_key = jax.random.split(key, 3)
            batch_size = real_imgs.shape[0]

            # 1. Sample random timesteps
            t = jax.random.randint(t_key, (batch_size,), 0, schedule.T)

            # 2. Sample Gaussian noise to add to the images
            epsilon = jax.random.normal(noise_key, real_imgs.shape)

            # 3. Drop the label on a fraction of the batch so the same network
            # also learns the unconditional score needed by guided sampling
            if num_classes is None:
                y = None
            else:
                dropped = jax.random.uniform(drop_key, (batch_size,)) < p_uncond
                y = jnp.where(dropped, num_classes, labels.astype(jnp.int32))

            # 4. MSE on the noise
            return jnp.mean(
                noise_prediction_error(unet, schedule, params, real_imgs, t, epsilon, y)
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.model_state.params, key)
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, state.model_state.params
        )
        new_params = optax.apply_updates(state.model_state.params, updates)

        metrics = {
            "loss": loss,
            "grad_norm": get_global_norm(grads),
            "param_norm": get_global_norm(new_params),
        }

        return state._replace(
            model_state=state.model_state._replace(params=new_params),
            opt_state=new_opt_state,
            step=state.step + 1,
            ema_params=ema_update(state.ema_params, new_params, state.step),
        ), metrics

    return init_train_state, train_step


def make_eval_fn(
    unet: hk.Transformed,
    schedule: DiffusionSchedule,
    val_batches,
    key: jax.Array,
    timesteps,
    num_classes: int | None = None,
):
    ts = jnp.asarray(timesteps, dtype=jnp.int32)
    n_t = ts.shape[0]

    @jax.jit
    def eval_batch(params, imgs, labels, batch_key):
        batch_size = imgs.shape[0]
        x_0 = jnp.tile(imgs, (n_t, 1, 1, 1))
        t = jnp.repeat(ts, batch_size)
        epsilon = jax.random.normal(batch_key, x_0.shape)
        y = None if num_classes is None else jnp.tile(labels.astype(jnp.int32), n_t)
        error = noise_prediction_error(unet, schedule, params, x_0, t, epsilon, y)
        return error.reshape(n_t, -1).mean(axis=1)

    def eval_fn(epoch, train_state):
        params = train_state.model_state.params
        per_batch = [
            eval_batch(params, x, y, jax.random.fold_in(key, i))
            for i, (x, y) in enumerate(val_batches)
        ]
        per_t = np.mean(jax.device_get(per_batch), axis=0)
        metrics = {"val_mse": float(per_t.mean())}
        metrics.update(
            {f"val_mse_t{int(t)}": float(v) for t, v in zip(timesteps, per_t)}
        )
        return metrics

    return eval_fn


def train_epoch(
    base_key: jax.Array,
    train_state: TrainState,
    train_step_fn,
    batches,
    steps: int,
    epoch: int,
    n_epochs: int,
    lr_schedule,
    print_every: int = 100,
):

    epoch_metrics: dict[str, list[float]] = {
        "loss": [],
        "grad_norm": [],
        "param_norm": [],
    }

    last_idx = 0
    data_wait = 0.0
    for i in range(steps):
        wait_start = time.perf_counter()
        try:
            batch = next(batches)
        except StopIteration:
            raise RuntimeError(
                f"Batch stream ended after {i} of {steps} steps in epoch {epoch}. "
                "steps_per_epoch does not match the data being passed in."
            ) from None
        data_wait += time.perf_counter() - wait_start
        # (images, labels) or images only
        if isinstance(batch, tuple):
            real_imgs, labels = batch[0], batch[1]
        else:
            real_imgs, labels = batch, None

        step_key = jax.random.fold_in(base_key, train_state.step)
        train_state, metrics = train_step_fn(step_key, train_state, real_imgs, labels)

        for k, v in metrics.items():
            epoch_metrics[k].append(float(v))

        if i > 0 and i % print_every == 0:
            lr = lr_schedule(int(train_state.step))
            win_loss = np.mean(epoch_metrics["loss"][last_idx:])
            win_grad = np.mean(epoch_metrics["grad_norm"][last_idx:])

            print(
                f"[Ep {epoch}/{n_epochs}][B {i}/{steps}] "
                f"Loss: {win_loss:.4f} | "
                f"Grad Norm: {win_grad:.2e} | "
                f"LR: {lr:.2e}"
            )
            last_idx = len(epoch_metrics["loss"])

    print(f"[Ep {epoch}/{n_epochs}] waited {format_duration(data_wait)} for data")
    return train_state, epoch_metrics


def fit_model(
    key: jax.Array,
    train_state: TrainState,
    train_step_fn,
    gen_fn,
    lr_schedule,
    train_ds,
    steps_per_epoch: int,
    n_epochs: int = 50,
    print_every: int = 100,
    figures_dir: str = "runs/figures",
    *,
    history: dict | None = None,
    checkpoint_fn=None,
    eval_fn=None,
):
    if history is None:
        history = {"loss": [], "grad_norm": [], "param_norm": []}

    train_key, sample_key = jax.random.split(key)

    start_step = int(train_state.step)
    batches = iter(train_ds)
    if start_step:
        if not hasattr(batches, "set_state"):
            raise TypeError(
                f"Cannot resume at step {start_step}: this batch stream cannot be "
                "positioned. Resuming reads the position from the step count, so "
                "the stream has to expose `set_state`, as grain's iterators do."
            )
        batches.set_state({"next_index": start_step})

    start_epoch = start_step // steps_per_epoch + 1

    print(f"\nStarting Training | Epochs: {n_epochs} | Batches: {steps_per_epoch}")
    print("-" * 80)

    run_start = time.time()
    for epoch in range(start_epoch, n_epochs + 1):
        start_time = time.time()

        train_state, epoch_data = train_epoch(
            base_key=train_key,
            train_state=train_state,
            train_step_fn=train_step_fn,
            batches=batches,
            steps=steps_per_epoch,
            epoch=epoch,
            n_epochs=n_epochs,
            lr_schedule=lr_schedule,
            print_every=print_every,
        )

        for key_name, values in epoch_data.items():
            history.setdefault(key_name, []).extend(values)

        epoch_time = time.time() - start_time

        os.makedirs(figures_dir, exist_ok=True)
        gen_fn(
            key=jax.random.fold_in(sample_key, epoch),
            model_state=train_state.model_state._replace(params=train_state.ema_params),
            filepath=f"{figures_dir}/epoch_{epoch}.png",
        )

        avg_loss = float(np.mean(epoch_data["loss"]))
        record = {"epoch": epoch, "train_loss": avg_loss}

        val_msg = ""
        if eval_fn is not None:
            eval_start = time.time()
            val_metrics = eval_fn(epoch, train_state)
            record.update(val_metrics)
            if "val_mse" in val_metrics:
                val_msg = (
                    f" | Val MSE: {val_metrics['val_mse']:.4f} "
                    f"({format_duration(time.time() - eval_start)})"
                )
        history.setdefault("epochs", []).append(record)

        print(
            f"[EPOCH {epoch}] COMPLETED | Time: {epoch_time:.1f}s | "
            f"AVG Loss: {avg_loss:.4f}{val_msg}"
        )

        if checkpoint_fn is not None:
            checkpoint_fn(epoch, train_state, history, epoch == n_epochs)

        elapsed = time.time() - run_start
        eta = elapsed / (epoch - start_epoch + 1) * (n_epochs - epoch)
        print(
            f"[EPOCH {epoch}] wall {format_duration(time.time() - start_time)} | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | "
            f"peak GPU memory {peak_memory()}"
        )
        print("-" * 80)

    return train_state, history
