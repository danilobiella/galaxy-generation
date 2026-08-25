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

if TYPE_CHECKING:
    from diffusion import DiffusionSchedule, Samplers


class ModelState(NamedTuple):
    params: hk.Params


class TrainState(NamedTuple):
    model_state: ModelState
    opt_state: optax.OptState
    step: jnp.ndarray


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


def make_train_fns(
    unet: hk.Transformed,
    optimizer: optax.GradientTransformation,
    schedule: DiffusionSchedule,
    num_classes: int | None = None,
    p_uncond: float = 0.1,
):
    def init_train_state(model_state: ModelState) -> TrainState:
        opt_state = optimizer.init(model_state.params)
        return TrainState(
            model_state=model_state,
            opt_state=opt_state,
            step=jnp.array(0, dtype=jnp.int32),
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

            # 3. Fetch cumulative alpha and reshape for broadcasting (B, 1, 1, 1)
            a_hat_t = schedule.alpha_hat[t][:, None, None, None]

            # 4. Forward process: Corrupt images to timestep t (Reparameterization Trick)
            x_t = jnp.sqrt(a_hat_t) * real_imgs + jnp.sqrt(1.0 - a_hat_t) * epsilon

            # 5. Drop the label on a fraction of the batch so the same network
            # also learns the unconditional score needed by guided sampling
            if num_classes is None:
                y = None
            else:
                dropped = jax.random.uniform(drop_key, (batch_size,)) < p_uncond
                y = jnp.where(dropped, num_classes, labels.astype(jnp.int32))

            # 6. Predict the added noise using the U-Net
            input = (x_t, t, y)
            eps_hat = unet.apply(params, None, input)

            # 7. Simple MSE Loss
            return jnp.mean(jnp.square(epsilon - eps_hat))

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
        ), metrics

    return init_train_state, train_step


def train_epoch(
    key: jax.Array,
    train_state: TrainState,
    train_step_fn,
    train_ds,
    epoch: int,
    n_epochs: int,
    lr_schedule,
    print_every: int = 100,
):
    n_batches = len(train_ds)
    ds_iter = iter(train_ds)

    epoch_metrics: dict[str, list[float]] = {
        "loss": [],
        "grad_norm": [],
        "param_norm": [],
    }

    last_idx = 0
    for i in range(n_batches):
        batch = next(ds_iter)
        # Extract images from tuple if your dataloader returns (images, labels)
        if isinstance(batch, tuple):
            real_imgs, labels = batch[0], batch[1]
        else:
            real_imgs, labels = batch, None

        key, step_key = jax.random.split(key)
        train_state, metrics = train_step_fn(step_key, train_state, real_imgs, labels)

        for k, v in metrics.items():
            epoch_metrics[k].append(float(v))

        if i > 0 and i % print_every == 0:
            lr = lr_schedule(int(train_state.step))
            win_loss = np.mean(epoch_metrics["loss"][last_idx:])
            win_grad = np.mean(epoch_metrics["grad_norm"][last_idx:])

            print(
                f"[Ep {epoch}/{n_epochs}][B {i}/{n_batches}] "
                f"Loss: {win_loss:.4f} | "
                f"Grad Norm: {win_grad:.2e} | "
                f"LR: {lr:.2e}"
            )
            last_idx = len(epoch_metrics["loss"])

    return train_state, epoch_metrics, key


def fit_model(
    key: jax.Array,
    train_state: TrainState,
    train_step_fn,
    gen_fn,
    lr_schedule,
    train_ds,
    n_epochs: int = 50,
    print_every: int = 100,
    figures_dir: str = "output/figures",
):
    history: dict[str, list[float]] = {
        "loss": [],
        "grad_norm": [],
        "param_norm": [],
    }

    print(f"\nStarting Training | Epochs: {n_epochs} | Batches: {len(train_ds)}")
    print("-" * 80)

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        key, subkey = jax.random.split(key)
        train_state, epoch_data, key = train_epoch(
            key=subkey,
            train_state=train_state,
            train_step_fn=train_step_fn,
            train_ds=train_ds,
            epoch=epoch,
            n_epochs=n_epochs,
            lr_schedule=lr_schedule,
            print_every=print_every,
        )

        for key_name, values in epoch_data.items():
            history[key_name].extend(values)

        epoch_time = time.time() - start_time

        os.makedirs(figures_dir, exist_ok=True)
        gen_fn(
            key=key,
            model_state=train_state.model_state,
            filepath=f"{figures_dir}/epoch_{epoch}.png",
        )

        avg_loss = np.mean(epoch_data["loss"])
        print(
            f"[EPOCH {epoch}] COMPLETED | Time: {epoch_time:.1f}s | "
            f"AVG Loss: {avg_loss:.4f}"
        )
        print("-" * 80)

    return train_state, history
