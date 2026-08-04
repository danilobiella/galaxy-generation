import os
import time
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore

from src.config import FIGURES_DIR


class ModelState(NamedTuple):
    params: hk.Params


class TrainState(NamedTuple):
    model_state: ModelState
    opt_state: optax.OptState
    step: jnp.ndarray


class DiffusionSchedule(NamedTuple):
    T: int
    beta: jnp.ndarray
    alpha: jnp.ndarray
    alpha_hat: jnp.ndarray

    @classmethod
    def create(cls, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        # Linear variance schedule
        beta = jnp.linspace(beta_start, beta_end, T)
        alpha = 1.0 - beta
        alpha_hat = jnp.cumprod(alpha)
        return cls(T=T, beta=beta, alpha=alpha, alpha_hat=alpha_hat)


@jax.jit
def get_global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))


def make_model_fns(
    unet: hk.Transformed,
    schedule: DiffusionSchedule,
    image_shape: tuple[int, int, int] = (256, 256, 3),
):
    def init_model_state(rng: jax.Array) -> ModelState:
        dummy_image = jnp.zeros((1, *image_shape))
        dummy_t = jnp.zeros((1,), dtype=jnp.int32)

        dummy_input = (dummy_image, dummy_t)
        params = unet.init(rng, dummy_input)
        return ModelState(params=params)

    def generate_and_save_images(
        key: jax.Array,
        model_state: ModelState,
        filepath: str = "generated_images.png",
    ):
        n_rows, n_cols = 4, 4
        n_samples = n_rows * n_cols

        @jax.jit
        def generate_batch(start_key, params):
            # 1. Start with pure Gaussian noise: x_T
            init_key, loop_key = jax.random.split(start_key)
            x_T = jax.random.normal(init_key, (n_samples, *image_shape))

            # 2. Reverse process loop body
            def body_fn(i, loop_state):
                x, step_key = loop_state

                # t goes from T-1 down to 0
                t = schedule.T - 1 - i
                t_batch = jnp.full((n_samples,), t, dtype=jnp.int32)

                # Predict noise added to this step
                input=(x, t_batch)
                eps_hat = unet.apply(params, None, input)

                # Extract scheduled values for current timestep
                a_t = schedule.alpha[t]
                a_hat_t = schedule.alpha_hat[t]
                b_t = schedule.beta[t]

                # Compute the mean of the previous step
                mu = (x - (b_t / jnp.sqrt(1.0 - a_hat_t)) * eps_hat) / jnp.sqrt(a_t)

                # Add variance (unless we are at the final step t=0)
                step_key, noise_key = jax.random.split(step_key)
                z = jax.random.normal(noise_key, x.shape)
                z = jnp.where(t > 0, z, jnp.zeros_like(z))

                x_prev = mu + jnp.sqrt(b_t) * z
                return x_prev, step_key

            # Run the loop efficiently on device
            x_0, _ = jax.lax.fori_loop(0, schedule.T, body_fn, (x_T, loop_key))
            return x_0

        generated_images = generate_batch(key, model_state.params)

        # Scale back from [-1, 1] to [0, 1] for plotting
        generated_images = jnp.clip((generated_images + 1.0) / 2.0, 0.0, 1.0)
        images_np = jax.device_get(generated_images)

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
):
    def init_train_state(model_state: ModelState) -> TrainState:
        opt_state = optimizer.init(model_state.params)
        return TrainState(
            model_state=model_state,
            opt_state=opt_state,
            step=jnp.array(0, dtype=jnp.int32),
        )

    @jax.jit
    def train_step(key, state, real_imgs):
        def loss_fn(params, key):
            t_key, noise_key = jax.random.split(key)
            batch_size = real_imgs.shape[0]

            # 1. Sample random timesteps
            t = jax.random.randint(t_key, (batch_size,), 0, schedule.T)

            # 2. Sample Gaussian noise to add to the images
            epsilon = jax.random.normal(noise_key, real_imgs.shape)

            # 3. Fetch cumulative alpha and reshape for broadcasting (B, 1, 1, 1)
            a_hat_t = schedule.alpha_hat[t][:, None, None, None]

            # 4. Forward process: Corrupt images to timestep t (Reparameterization Trick)
            x_t = jnp.sqrt(a_hat_t) * real_imgs + jnp.sqrt(1.0 - a_hat_t) * epsilon

            # 5. Predict the added noise using the U-Net
            input = (x_t, t)
            eps_hat = unet.apply(params, None, input)

            # 6. Simple MSE Loss
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
        real_imgs = batch[0] if isinstance(batch, tuple) else batch

        key, step_key = jax.random.split(key)
        train_state, metrics = train_step_fn(step_key, train_state, real_imgs)

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
    figures_dir: str = str(FIGURES_DIR),
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
