import matplotlib.pyplot as plt

from src.data_loader import make_data_loaders

BATCH_SIZE = 16
SEED = 42
DATA_PATH = "data/galaxy10_decals_lmdb"
IMAGE_SIZE = (256, 256)

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

it = iter(train_data_loader)
batch = next(it)

images_np = batch["image"]
images_np = (images_np + 1.0) / 2.0

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i in range(16):
    ax = axes[i // 4, i % 4]
    img = images_np[i]
    if img.shape[-1] == 1:
        ax.imshow(img.squeeze(), cmap="gray")
    else:
        ax.imshow(img)

    ax.axis("off")
plt.savefig("example_images.png", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
