
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from src.data_loader import make_data_loaders
from src.models import build_network
from src.train import fit_model, make_model_fns, make_train_fns

BATCH_SIZE = 16
N_EPOCHS = 50
SEED = 42
DATA_PATH = "data/galaxy10_decals_lmdb"
IMAGE_SIZE = (256, 256)
LATENT_DIM = 128
N_GEN_STEPS = 10
N_DIS_STEPS = 10

# KAGGLE_BASE_PATHS = [
#     "/kaggle/input/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0/train",
# ]
# LOCAL_BASE_PATHS = [
#     "data/zero/train",
#     "data/one/train",
# ]
#
# if Path("/kaggle/input").exists():
#     print("Running on Kaggle")
#     images_paths = KAGGLE_BASE_PATHS
# else:
#     print("Running locally")
#     images_paths = LOCAL_BASE_PATHS

# %% -- data ---------------------------------------------------------------


train_data_loader, val_data_loader = make_data_loaders(
    db_path=DATA_PATH, batch_size=BATCH_SIZE, val_split=0.1, seed=SEED
)

steps_per_epoch = len(train_data_loader)
n_steps = N_EPOCHS * steps_per_epoch

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {n_steps}")

print("Made train and val data loaders")
print(f"Numer of training batches: {len(train_data_loader)}")
print(f"Number of validation batches: {len(val_data_loader)}")

print(f"Number of training samples: {len(train_data_loader) * BATCH_SIZE}")
print(f"Number of validation samples: {len(val_data_loader) * BATCH_SIZE}")

# %%

gen, dis = build_network()
print("built network")

print("Discriminator architecture:")
print(
    hk.experimental.tabulate(
        dis, columns=["module", "input", "output", "params_size", "params_bytes"]
    )(jnp.zeros((BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
)
print("\nGenerator architecture:")
print(
    hk.experimental.tabulate(
        gen, columns=["module", "input", "output", "params_size", "params_bytes"]
    )(jnp.zeros((BATCH_SIZE, LATENT_DIM)))
)


def schedule(step):
    return 1e-4


optimizer = optax.adam(learning_rate=1e-4, b1=0.5)
print("built optimizer and schedule")

# %%

init_model_state, eval_step, generate_fn = make_model_fns(
    gen, dis, latent_dim=LATENT_DIM
)
init_train_state, gen_train_step, dis_train_step = make_train_fns(
    gen, dis, optimizer, latent_dim=LATENT_DIM
)

example_batch = next(iter(train_data_loader))
input_shape = example_batch["image"].shape

model_state = init_model_state(
    rng=jax.random.PRNGKey(SEED),
    image_shape=input_shape[1:],
)

train_state = init_train_state(model_state)
print("Initialized model and training state")

# key = jax.random.PRNGKey(SEED)
# gen_train_step(key, train_state, 16)
#
# dis_train_step(key, train_state, example_batch["image"])

# %% -- train --------------------------------------------------------------

key = jax.random.PRNGKey(SEED)
state, history = fit_model(
    key=key,
    train_state=train_state,
    gen_step=gen_train_step,
    dis_step=dis_train_step,
    gen_fn=generate_fn,
    schedule=schedule,
    train_ds=train_data_loader,
    val_ds=val_data_loader,
    n_epochs=N_EPOCHS,
    n_gen_steps=N_GEN_STEPS,
    n_dis_steps=N_DIS_STEPS,
)

# # -- plots --------------------------------------------------------------
# plt.figure(figsize=(8, 4))
# plt.plot(history["train_loss"], label="train_loss")
# plt.plot(history["val_loss"], label="val_loss")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()
#
# plt.figure(figsize=(8, 4))
# plt.plot(history["train_accuracy"], label="train_accuracy")
# plt.plot(history["val_accuracy"], label="val_accuracy")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()
#
# # %% save network state and history

import pickle

with open("state.pkl", "wb") as f:
    pickle.dump(state, f)

with open("history.pkl", "wb") as f:
    pickle.dump(history, f)
#
# # %%
#
# # -- test ---------------------------------------------------------------
# x_test, y_test = load_cifar_test(BASE_PATH)
#
# test_ds = make_grain_iterator(
#     x_test,
#     y_test,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     seed=0,
#     num_epochs=None,
#     drop_remainder=True,
#     augment=False,
# )
#
# n_test_batches = len(x_test) // BATCH_SIZE
#
# test_metrics = evaluate_dataset(
#     params=state.params,
#     bn_state=state.bn_state,
#     eval_step=eval_step,
#     val_ds=test_ds,
#     n_batches=n_test_batches,
# )
#
# print(f"Test loss: {test_metrics['loss']:.4f}")
# print(f"Test Error: {1 - test_metrics['accuracy']:.4f}")
