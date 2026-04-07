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
import rich
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)


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
            c_val = (class_idx - 5.0) / 5.0

            key, col_key = jax.random.split(key)
            z = jax.random.normal(col_key, (n_rows, 128))
            input = (z, jnp.full((n_rows,), c_val))

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
    """
    Training API.

    Returns:
        init_train_state: create optimizer state from a ModelState
        train_step: one SGD/optimizer update step
    """

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
    ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        def gen_loss(gen_params, dis_params, key):
            z_key, c_key = jax.random.split(key)
            z = jax.random.normal(key, (batch_size, latent_dim))
            c = jax.random.randint(c_key, (batch_size,), 0, 10)
            c = (c - 5.0) / 5.0
            gen_input = (z, c)
            gen_images, new_gen_nn_state = gen.apply(
                gen_params,
                state.model_state.gen_nn_state,
                None,
                gen_input,
                is_training=True,
            )
            dis_input = (gen_images, c)
            logits, _ = dis.apply(
                dis_params,
                state.model_state.dis_nn_state,
                None,
                dis_input,
                is_training=True,
            )
            loss = jnp.mean(jax.nn.softplus(-logits))
            return loss, new_gen_nn_state

        key, step_key = jax.random.split(key)
        (gen_loss, gen_nn_state), gen_grads = jax.value_and_grad(
            gen_loss, argnums=0, has_aux=True
        )(state.model_state.gen_params, state.model_state.dis_params, step_key)

        gen_updates, new_gen_opt_state = optimizer.update(
            gen_grads,
            state.gen_opt_state,
            state.model_state.gen_params,
        )

        new_gen_params = optax.apply_updates(state.model_state.gen_params, gen_updates)

        new_state = TrainState(
            model_state=ModelState(
                gen_params=new_gen_params,
                gen_nn_state=gen_nn_state,
                dis_params=state.model_state.dis_params,
                dis_nn_state=state.model_state.dis_nn_state,
            ),
            gen_opt_state=new_gen_opt_state,
            dis_opt_state=state.dis_opt_state,
            step=state.step + 1,
        )

        return new_state, {"gen_loss": gen_loss}  # type: ignore

    @jax.jit
    def dis_train_step(
        key: jax.Array,
        state: TrainState,
        real_input: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        def dis_loss(gen_params, disc_params, key):
            key_z, key_c = jax.random.split(key)
            z = jax.random.normal(key, (real_input[0].shape[0], latent_dim))
            c = jax.random.randint(key_c, (real_input[1].shape[0],), 0, 10)
            c = (c - 5.0) / 5.0
            gen_input = (z, c)
            gen_images, _ = gen.apply(
                gen_params,
                state.model_state.gen_nn_state,
                None,
                gen_input,
                is_training=True,
            )
            real_logits, new_nn_state = dis.apply(
                disc_params,
                state.model_state.dis_nn_state,
                None,
                real_input,
                is_training=True,
            )
            disc_input = (gen_images, c)
            fake_logits, new_nn_state = dis.apply(
                disc_params,
                new_nn_state,
                None,
                disc_input,
                is_training=True,
            )

            real_loss = jax.nn.softplus(-real_logits)
            fake_loss = jax.nn.softplus(fake_logits)
            loss = jnp.mean(real_loss + fake_loss)

            return loss, new_nn_state

        key, step_key = jax.random.split(key)
        (dis_loss, dis_nn_state), dis_grads = jax.value_and_grad(
            dis_loss, argnums=1, has_aux=True
        )(state.model_state.gen_params, state.model_state.dis_params, step_key)

        dis_updates, new_dis_opt_state = optimizer.update(
            dis_grads,
            state.dis_opt_state,
            state.model_state.dis_params,
        )

        new_dis_params = optax.apply_updates(state.model_state.dis_params, dis_updates)

        new_state = TrainState(
            model_state=ModelState(
                gen_params=state.model_state.gen_params,
                gen_nn_state=state.model_state.gen_nn_state,
                dis_params=new_dis_params,
                dis_nn_state=dis_nn_state,
            ),
            gen_opt_state=state.gen_opt_state,
            dis_opt_state=new_dis_opt_state,
            step=state.step + 1,
        )

        return new_state, {"dis_loss": dis_loss}  # type: ignore

    return init_train_state, gen_train_step, dis_train_step


def evaluate_dataset(model_state, eval_step, data_loader, progress_bar=False):
    total_correct = 0
    total_top5 = 0
    total_samples = 0
    losses = []

    if progress_bar:
        data_loader = rich.progress.track(
            data_loader, description="[bold cyan]Evaluating...", transient=True
        )
    for batch in data_loader:
        xb, yb = batch["x"], batch["y"]
        metrics = eval_step(model_state, xb, yb)

        preds = jnp.argmax(metrics["logits"], axis=-1)
        correct = jnp.sum(preds == yb)

        top5 = jax.lax.top_k(metrics["logits"], 5)[1]
        top5_correct = jnp.sum(jnp.any(top5 == yb[:, None], axis=1))

        batch_size = yb.shape[0]

        total_correct += int(correct)
        total_top5 += int(top5_correct)
        total_samples += batch_size

        losses.append(float(metrics["loss"]))

    return {
        "loss": float(np.mean(losses)),
        "accuracy": total_correct / total_samples,
        "top5_accuracy": total_top5 / total_samples,
    }


# --- Example Usage ---
# To integrate this into your existing loop (e.g., at the end of fit_model):
#
# generate_and_save_images(
#     key=key,                   # Pass current JAX random key
#     gen=gen,                   # The hk.TransformedWithState generator object
#     model_state=train_state.model_state, # Current parameters and NN state
#     latent_dim=LATENT_DIM,      # e.g., 10 or 128
#     filepath=f"epoch_{epoch}_samples.png"
# )


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
    console: Console,
):
    gen_losses = []
    dis_losses = []

    n_batches = len(train_ds)

    progress = Progress(
        TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.fields[n_epochs]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("gen_loss: {task.fields[gen_loss]:.4f}"),
        TextColumn("•"),
        TextColumn("dis_loss: {task.fields[dis_loss]:.4f}"),
        TextColumn("•"),
        TextColumn("lr: {task.fields[lr]:.2e}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )

    with progress:
        task_id = progress.add_task(
            "train",
            total=n_batches,
            epoch=epoch,
            n_epochs=n_epochs,
            gen_loss=0.0,
            dis_loss=0.0,
            lr=float(schedule(int(train_state.step))),
        )

        batch_loading_time = []
        train_batch_time = []

        ds_iter = iter(train_ds)

        for i in range(n_batches):
            st = time.time()
            batch = next(ds_iter)
            batch_loading_time.append(time.time() - st)

            st = time.time()
            key, step_key = jax.random.split(key)
            train_state, metrics = dis_step(step_key, train_state, batch)
            train_batch_time.append(time.time() - st)

            dis_batch_loss = float(metrics["dis_loss"])
            dis_losses.append(dis_batch_loss)

            if i % n_dis_steps == 0:
                for _ in range(n_gen_steps):
                    key, step_key = jax.random.split(key)
                    train_state, metrics = gen_step(
                        step_key, train_state, batch[0].shape[0]
                    )
                    gen_batch_loss = float(metrics["gen_loss"])
                    gen_losses.append(gen_batch_loss)

            progress.update(
                task_id,
                advance=1,
                dis_loss=float(np.mean(dis_losses)),
                gen_loss=float(np.mean(gen_losses)),
                lr=float(schedule(int(train_state.step))),
            )

    train_metrics = {
        "gen_loss": float(np.mean(gen_losses)),
        "dis_loss": float(np.mean(dis_losses)),
        "batch_loading_time": float(np.mean(batch_loading_time)),
        "train_batch_time": float(np.mean(train_batch_time)),
    }

    return train_state, train_metrics, key


def fit_model(
    key: jax.Array,
    train_state: TrainState,
    gen_step,
    dis_step,
    gen_fn,
    schedule,
    train_ds,
    n_epochs: int = 20,
    n_gen_steps: int = 10,
    n_dis_steps: int = 10,
):
    print("Starting training...")
    console = Console()

    history: dict[str, list[float]] = {
        "gen_train_loss": [],
        "dis_train_loss": [],
        "lr": [],
    }

    console.print(
        f"[bold blue]Training setup[/bold blue]  train_batches={len(train_ds)}  "
    )

    key, gen_key = jax.random.split(key)
    for epoch in range(1, n_epochs + 1):
        key, subkey = jax.random.split(key)
        train_state, train_metrics, key = train_epoch(
            key=subkey,
            train_state=train_state,
            gen_step=gen_step,
            dis_step=dis_step,
            train_ds=train_ds,
            epoch=epoch,
            n_gen_steps=n_gen_steps,
            n_dis_steps=n_dis_steps,
            n_epochs=n_epochs,
            schedule=schedule,
            console=console,
        )

        # Generate and save sample images at the end of each epoch
        os.makedirs("figures", exist_ok=True)
        gen_fn(
            key=gen_key,
            model_state=train_state.model_state,
            filepath=f"figures/epoch_{epoch}_samples.png",
        )

        lr = float(schedule(int(train_state.step)))

        history["gen_train_loss"].append(train_metrics["gen_loss"])
        history["dis_train_loss"].append(train_metrics["dis_loss"])
        history["lr"].append(lr)

        console.print(
            f"[bold green]Epoch {epoch:02d}/{n_epochs}[/bold green]  "
            f"gen_train_loss={train_metrics['gen_loss']:.4f}  "
            f"dis_train_loss={train_metrics['dis_loss']:.4f}  "
            f"batch_loading_time={train_metrics['batch_loading_time']:.4f}  "
            f"train_batch_time={train_metrics['train_batch_time']:.4f}  "
            f"lr={lr:.2e}",
        )

    return train_state, history
