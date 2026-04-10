import os
import time
from functools import partial
from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore


class ModelState(NamedTuple):
    gen_params: hk.Params
    gen_nn_state: hk.State
    dis_params: hk.Params
    dis_nn_state: hk.State


class TrainState(NamedTuple):
    model_state: ModelState
    gen_opt_state: optax.OptState
    dis_opt_state: optax.OptState
    step: jnp.ndarray


IDX_TO_CLASS = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth\nGalaxies",
    3: "In-between Round\nSmooth Galaxies",
    4: "Cigar Shaped\nSmooth Galaxies",
    5: "Barred Spiral\nGalaxies",
    6: "Unbarred Tight\nSpiral Galaxies",
    7: "Unbarred Loose\nSpiral Galaxies",
    8: "Edge-on Galaxies\nwithout Bulge",
    9: "Edge-on Galaxies\nwith Bulge",
}


@jax.jit
def get_global_norm(tree):
    """Calculates the L2 norm of a pytree (params or grads)."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))


def make_model_fns(
    gen: hk.TransformedWithState,
    dis: hk.TransformedWithState,
    latent_dim: int = 128,
    image_shape: tuple[int, int, int] = (256, 256, 3),
):
    """
    Model-only API.

    Returns:
        init_model_state: initialize params + nn_state
        gen_fn: generate and save images from the current generator state
    """

    def init_model_state(
        rng: jax.Array,
    ) -> ModelState:
        dummy_image = jnp.zeros((1, *image_shape))  # [1, H, W, C]
        dummy_z = jnp.zeros((1, latent_dim))  # [1, latent_dim]
        dummy_label = jnp.zeros((1,), dtype=jnp.int32)  # [1] class label
        dummy_image_input = (dummy_image, dummy_label)
        dummy_gen_input = (dummy_z, dummy_label)
        dis_params, dis_nn_state = dis.init(rng, dummy_image_input, is_training=True)
        gen_params, gen_nn_state = gen.init(rng, dummy_gen_input, is_training=True)
        return ModelState(
            gen_params=gen_params,
            gen_nn_state=gen_nn_state,
            dis_params=dis_params,
            dis_nn_state=dis_nn_state,
        )

    def generate_and_save_images(
        key: jax.Array,
        model_state: ModelState,
        filepath: str = "generated_images.png",
    ):
        """
        Generates a 5x10 grid: 10 columns (classes) and 5 rows (samples per class).
        """
        n_rows = 5
        n_cols = 10

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        @jax.jit
        def _generate(params, state, input):
            images, _ = gen.apply(params, state, None, input, is_training=False)
            return images

        for class_idx in range(n_cols):
            key, col_key = jax.random.split(key)
            z = jax.random.normal(col_key, (n_rows, 128))
            input = (z, jnp.full((n_rows,), class_idx))

            generated_images = _generate(
                model_state.gen_params, model_state.gen_nn_state, input
            )

            generated_images = (generated_images + 1.0) / 2.0
            images_np = jax.device_get(generated_images)

            class_name = IDX_TO_CLASS.get(class_idx, f"Class {class_idx}")
            axes[0, class_idx].set_title(class_name, fontsize=10, fontweight="bold")

            for row_idx in range(n_rows):
                ax = axes[row_idx, class_idx]
                img = images_np[row_idx]

                if img.shape[-1] == 1:
                    ax.imshow(img.squeeze(), cmap="gray")
                else:
                    ax.imshow(img)

                ax.axis("off")

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        print(f"Generated samples saved to: {filepath}")

    return init_model_state, generate_and_save_images


def make_train_fns(
    gen: hk.TransformedWithState,
    dis: hk.TransformedWithState,
    optimizer: optax.GradientTransformation,
    latent_dim: int = 128,
):
    def init_train_state(model_state: ModelState) -> TrainState:
        gen_opt_state = optimizer.init(model_state.gen_params)
        dis_opt_state = optimizer.init(model_state.dis_params)
        return TrainState(
            model_state=model_state,
            gen_opt_state=gen_opt_state,
            dis_opt_state=dis_opt_state,
            step=jnp.array(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnames=["batch_size"])
    def gen_train_step(
        key: jax.Array,
        state: TrainState,
        batch_size: int,
    ) -> tuple[TrainState, dict]:
        def gen_loss_fn(gen_params, dis_params, key):
            z_key, c_key = jax.random.split(key)
            z = jax.random.normal(z_key, (batch_size, latent_dim))
            c = jax.random.randint(c_key, (batch_size,), 0, 10)

            gen_images, new_gen_nn_state = gen.apply(
                gen_params,
                state.model_state.gen_nn_state,
                None,
                (z, c),
                is_training=True,
            )
            logits, _ = dis.apply(
                dis_params,
                state.model_state.dis_nn_state,
                None,
                (gen_images, c),
                is_training=True,
            )
            loss = jnp.mean(jax.nn.softplus(-logits))
            return loss, new_gen_nn_state

        (loss, gen_nn_state), grads = jax.value_and_grad(gen_loss_fn, has_aux=True)(
            state.model_state.gen_params, state.model_state.dis_params, key
        )

        updates, new_opt_state = optimizer.update(
            grads, state.gen_opt_state, state.model_state.gen_params
        )
        new_params = optax.apply_updates(state.model_state.gen_params, updates)

        metrics = {
            "gen_loss": loss,
            "gen_grad_norm": get_global_norm(grads),
            "gen_param_norm": get_global_norm(new_params),
        }

        new_state = state._replace(
            model_state=state.model_state._replace(
                gen_params=new_params, gen_nn_state=gen_nn_state
            ),
            gen_opt_state=new_opt_state,
            step=state.step + 1,
        )
        return new_state, metrics

    @jax.jit
    def dis_train_step(
        key: jax.Array,
        state: TrainState,
        real_input: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[TrainState, dict]:
        def dis_loss_fn(disc_params, gen_params, key):
            z_key, c_key = jax.random.split(key)
            z = jax.random.normal(z_key, (real_input[0].shape[0], latent_dim))
            c = jax.random.randint(c_key, (real_input[1].shape[0],), 0, 10)

            fake_images, _ = gen.apply(
                gen_params,
                state.model_state.gen_nn_state,
                None,
                (z, c),
                is_training=True,
            )

            # Real forward
            real_logits, new_nn_state = dis.apply(
                disc_params,
                state.model_state.dis_nn_state,
                None,
                real_input,
                is_training=True,
            )
            # Fake forward
            fake_logits, new_nn_state = dis.apply(
                disc_params, new_nn_state, None, (fake_images, c), is_training=True
            )

            l_real = jnp.mean(
                0.9 * jax.nn.softplus(-real_logits) + 0.1 * jax.nn.softplus(real_logits)
            )
            l_fake = jnp.mean(jax.nn.softplus(fake_logits))

            total_loss = (l_real + l_fake) / 2.0
            return total_loss, (l_real, l_fake, new_nn_state)

        (loss, (l_real, l_fake, dis_nn_state)), grads = jax.value_and_grad(
            dis_loss_fn, has_aux=True
        )(state.model_state.dis_params, state.model_state.gen_params, key)

        updates, new_opt_state = optimizer.update(
            grads, state.dis_opt_state, state.model_state.dis_params
        )
        new_params = optax.apply_updates(state.model_state.dis_params, updates)

        metrics = {
            "dis_loss": loss,
            "dis_real_loss": l_real,
            "dis_fake_loss": l_fake,
            "dis_grad_norm": get_global_norm(grads),
            "dis_param_norm": get_global_norm(new_params),
        }

        new_state = state._replace(
            model_state=state.model_state._replace(
                dis_params=new_params, dis_nn_state=dis_nn_state
            ),
            dis_opt_state=new_opt_state,
            step=state.step + 1,
        )
        return new_state, metrics

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

    # Store EVERYTHING for high-res plotting later
    epoch_metrics = {
        "dis_loss": [],
        "dis_real_loss": [],
        "dis_fake_loss": [],
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
            win_real = np.mean(epoch_metrics["dis_real_loss"][last_d_idx:])
            win_fake = np.mean(epoch_metrics["dis_fake_loss"][last_d_idx:])
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
                f"D_L: {win_dis_loss:.4f} (R:{win_real:.3f}/F:{win_fake:.3f}) | "
                f"G_Grad: {win_g_grad:.2e} | D_Grad: {win_d_grad:.2e} | "
                f"LR: {lr:.2e}"
            )

            # Slide the window
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
    history = {
        "dis_loss": [],
        "dis_real_loss": [],
        "dis_fake_loss": [],
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
