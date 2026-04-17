import os
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore


class ModelState(NamedTuple):
    gen_params: hk.Params
    dis_params: hk.Params


class TrainState(NamedTuple):
    model_state: ModelState
    gen_opt_state: optax.OptState
    dis_opt_state: optax.OptState
    step: jnp.ndarray


@jax.jit
def get_global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))


def make_model_fns(
    gen: hk.Transformed,
    dis: hk.Transformed,
    latent_dim: int = 128,
    image_shape: tuple[int, int, int] = (256, 256, 3),
):
    def init_model_state(rng: jax.Array) -> ModelState:
        dummy_z = jnp.zeros((1, latent_dim))
        dummy_label = jnp.zeros((1,), dtype=jnp.int32)
        dummy_image = jnp.zeros((1, *image_shape))

        gen_params = gen.init(rng, (dummy_z, dummy_label))
        dis_params = dis.init(rng, (dummy_image, dummy_label))

        return ModelState(gen_params=gen_params, dis_params=dis_params)

    def generate_and_save_images(
        key: jax.Array,
        model_state: ModelState,
        filepath: str = "generated_images.png",
    ):
        n_rows, n_cols = 5, 10
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))

        @jax.jit
        def _generate(params, input):
            return gen.apply(params, None, input)

        for class_idx in range(n_cols):
            key, col_key = jax.random.split(key)
            z = jax.random.normal(col_key, (n_rows, latent_dim))
            input = (z, jnp.full((n_rows,), class_idx))

            generated_images = _generate(model_state.gen_params, input)
            generated_images = (generated_images + 1.0) / 2.0
            images_np = jax.device_get(generated_images)

            for row_idx in range(n_rows):
                ax = axes[row_idx, class_idx]
                ax.imshow(images_np[row_idx])
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated samples saved to: {filepath}")

    return init_model_state, generate_and_save_images


def make_train_fns(
    gen: hk.Transformed,
    dis: hk.Transformed,
    gen_optimizer: optax.GradientTransformation,
    dis_optimizer: optax.GradientTransformation,
    latent_dim: int = 128,
    lambda_gp: float = 20.0,
):
    def init_train_state(model_state: ModelState) -> TrainState:
        gen_opt_state = gen_optimizer.init(model_state.gen_params)
        dis_opt_state = dis_optimizer.init(model_state.dis_params)
        return TrainState(
            model_state=model_state,
            gen_opt_state=gen_opt_state,
            dis_opt_state=dis_opt_state,
            step=jnp.array(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnames=["batch_size"])
    def gen_train_step(key, state, batch_size):
        def gen_loss_fn(gen_params, dis_params, key):
            z_key, c_key = jax.random.split(key)
            z = jax.random.normal(z_key, (batch_size, latent_dim))
            c = jax.random.randint(c_key, (batch_size,), 0, 10)

            gen_images = gen.apply(gen_params, None, (z, c))
            logits = dis.apply(dis_params, None, (gen_images, c))
            return -jnp.mean(logits)  # WGAN Gen Loss

        loss, grads = jax.value_and_grad(gen_loss_fn)(
            state.model_state.gen_params, state.model_state.dis_params, key
        )
        updates, new_opt_state = gen_optimizer.update(
            grads, state.gen_opt_state, state.model_state.gen_params
        )
        new_gen_params = optax.apply_updates(state.model_state.gen_params, updates)

        return state._replace(
            model_state=state.model_state._replace(gen_params=new_gen_params),
            gen_opt_state=new_opt_state,
            step=state.step + 1,
        ), {
            "gen_loss": loss,
            "gen_grad_norm": get_global_norm(grads),
            "gen_param_norm": get_global_norm(new_gen_params),
        }

    @jax.jit
    def dis_train_step(key, state, real_input):
        real_imgs, labels = real_input

        def dis_loss_fn(dis_params, gen_params, key):
            key_z, key_gp = jax.random.split(key)
            z = jax.random.normal(key_z, (real_imgs.shape[0], latent_dim))
            fake_imgs = gen.apply(gen_params, None, (z, labels))

            real_logits = dis.apply(dis_params, None, (real_imgs, labels))
            fake_logits = dis.apply(dis_params, None, (fake_imgs, labels))

            w_loss = jnp.mean(fake_logits) - jnp.mean(real_logits)

            # --- Gradient Penalty Logic ---
            alpha = jax.random.uniform(key_gp, (real_imgs.shape[0], 1, 1, 1))
            interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs

            # 1. Correct the internal function to use its arguments (img, lbl)
            def critic_forward(img, lbl):
                # We add [None] to both to create a "batch of 1" for Haiku Conv layers
                logits = dis.apply(dis_params, None, (img[None], lbl[None]))
                return jnp.sum(logits)

            # 2. Tell vmap to slice BOTH arguments (in_axes=(0, 0))
            grad_fn = jax.vmap(jax.grad(critic_forward), in_axes=(0, 0))

            # 3. Pass both the interpolated images and the batch labels
            interp_grads = grad_fn(interpolates, labels)

            norms = jnp.sqrt(jnp.sum(jnp.square(interp_grads), axis=(1, 2, 3)) + 1e-8)
            gp = jnp.mean((norms - 1.0) ** 2)

            return w_loss + lambda_gp * gp, (w_loss, jnp.mean(norms))

        # The rest of the function remains the same...
        (total_loss, (w_loss, grad_norm)), grads = jax.value_and_grad(
            dis_loss_fn, has_aux=True
        )(state.model_state.dis_params, state.model_state.gen_params, key)

        updates, new_opt_state = dis_optimizer.update(
            grads, state.dis_opt_state, state.model_state.dis_params
        )
        new_dis_params = optax.apply_updates(state.model_state.dis_params, updates)

        metrics = {
            "dis_loss": total_loss,
            "w_loss": w_loss,
            "dis_grad_norm": grad_norm,
            "dis_param_norm": get_global_norm(new_dis_params),
        }

        return state._replace(
            model_state=state.model_state._replace(dis_params=new_dis_params),
            dis_opt_state=new_opt_state,
            step=state.step + 1,
        ), metrics

    return init_train_state, gen_train_step, dis_train_step


def train_epoch(
    key: jax.Array,
    train_state: TrainState,
    gen_step,
    dis_step,
    train_ds,
    epoch: int,
    n_epochs: int,
    n_gen_steps: int,
    n_dis_steps: int,
    schedule,
    print_every: int = 100,
):
    n_batches = len(train_ds)
    ds_iter = iter(train_ds)

    epoch_metrics: dict[str, list[float]] = {
        "dis_loss": [],
        "w_loss": [],
        "dis_grad_norm": [],
        "dis_param_norm": [],
        "gen_loss": [],
        "gen_grad_norm": [],
        "gen_param_norm": [],
    }

    # Tracking indices for windowed averaging
    last_d_idx = 0
    last_g_idx = 0
    for i in range(n_batches):
        batch = next(ds_iter)

        key, step_key = jax.random.split(key)
        train_state, d_metrics = dis_step(step_key, train_state, batch)

        for k, v in d_metrics.items():
            if k in epoch_metrics:
                epoch_metrics[k].append(float(v))

        if i % n_dis_steps == 0:
            for _ in range(n_gen_steps):
                key, step_key = jax.random.split(key)
                train_state, g_metrics = gen_step(
                    step_key, train_state, batch[0].shape[0]
                )
                for k, v in g_metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(float(v))

        if i > 0 and i % print_every == 0:
            lr = schedule(int(train_state.step))

            win_dis_loss = np.mean(epoch_metrics["dis_loss"][last_d_idx:])
            win_w_loss = np.mean(epoch_metrics["w_loss"][last_d_idx:])
            win_d_grad = np.mean(epoch_metrics["dis_grad_norm"][last_d_idx:])
            win_gen_loss = (
                np.mean(epoch_metrics["gen_loss"][last_g_idx:])
                if epoch_metrics["gen_loss"]
                else 0.0
            )
            win_g_grad = (
                np.mean(epoch_metrics["gen_grad_norm"][last_g_idx:])
                if epoch_metrics["gen_grad_norm"]
                else 0.0
            )

            print(
                f"[Ep {epoch}/{n_epochs}][B {i}/{n_batches}] "
                f"G_L: {win_gen_loss:.4f} | "
                f"D_L: {win_dis_loss:.4f} | W_D: {win_w_loss:.4f} | "
                f"G_Grad: {win_g_grad:.2e} | D_Grad: {win_d_grad:.2e} | "
                f"LR: {lr:.2e}"
            )

            last_d_idx = len(epoch_metrics["dis_loss"])
            last_g_idx = len(epoch_metrics["gen_loss"])

    return train_state, epoch_metrics, key


def fit_model(
    key: jax.Array,
    train_state: TrainState,
    gen_step,
    dis_step,
    gen_fn,
    schedule,
    train_ds,
    n_epochs: int = 50,
    n_gen_steps: int = 10,
    n_dis_steps: int = 10,
    print_every: int = 100,
):
    history: dict[str, list[float]] = {
        "dis_loss": [],
        "w_loss": [],
        "dis_grad_norm": [],
        "dis_param_norm": [],
        "gen_loss": [],
        "gen_grad_norm": [],
        "gen_param_norm": [],
    }

    print(f"\nStarting Training | Epochs: {n_epochs} | Batches: {len(train_ds)}")
    print("-" * 110)

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        key, subkey = jax.random.split(key)
        train_state, epoch_data, key = train_epoch(
            key=subkey,
            train_state=train_state,
            gen_step=gen_step,
            dis_step=dis_step,
            train_ds=train_ds,
            epoch=epoch,
            n_epochs=n_epochs,
            n_gen_steps=n_gen_steps,
            n_dis_steps=n_dis_steps,
            schedule=schedule,
            print_every=print_every,
        )

        for key_name, values in epoch_data.items():
            history[key_name].extend(values)

        epoch_time = time.time() - start_time

        os.makedirs("figures", exist_ok=True)
        gen_fn(
            key=key,
            model_state=train_state.model_state,
            filepath=f"figures/epoch_{epoch}.png",
        )

        avg_g = np.mean(epoch_data["gen_loss"]) if epoch_data["gen_loss"] else 0.0
        avg_d = np.mean(epoch_data["dis_loss"])
        print(
            f"[EPOCH {epoch}] COMPLETED | Time: {epoch_time:.1f}s | "
            f"AVG G_Loss: {avg_g:.4f} | AVG D_Loss: {avg_d:.4f}"
        )
        print("-" * 110)

    return train_state, history
