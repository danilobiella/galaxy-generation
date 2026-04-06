import haiku as hk
import jax
import jax.numpy as jnp


class GenResBlock(hk.Module):
    """Upsampling Residual Block for the Generator."""

    def __init__(self, out_channels, stride=2, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.stride = stride

    def __call__(self, x, is_training):
        in_channels = x.shape[-1]

        # --- Main Path ---
        # 1. Upsample convolution
        h = hk.Conv2DTranspose(
            self.out_channels, kernel_shape=4, stride=self.stride, padding="SAME"
        )(x)
        h = hk.BatchNorm(True, True, 0.9)(h, is_training)
        h = jax.nn.relu(h)

        # 2. Refinement convolution (stride 1)
        h = hk.Conv2D(self.out_channels, kernel_shape=3, stride=1, padding="SAME")(h)
        h = hk.BatchNorm(True, True, 0.9)(h, is_training)

        # --- Shortcut Path ---
        # We must project the shortcut to match the new resolution and channel depth
        shortcut = hk.Conv2DTranspose(
            self.out_channels, kernel_shape=1, stride=self.stride, padding="SAME"
        )(x)

        return jax.nn.relu(h + shortcut)


class DisResBlock(hk.Module):
    """Downsampling Residual Block for the Discriminator."""

    def __init__(self, out_channels, stride=2, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.stride = stride

    def __call__(self, x, is_training):
        in_channels = x.shape[-1]

        # --- Main Path ---
        h = hk.Conv2D(
            self.out_channels, kernel_shape=4, stride=self.stride, padding="SAME"
        )(x)
        h = hk.BatchNorm(True, True, 0.9)(h, is_training)
        h = jax.nn.leaky_relu(h, negative_slope=0.2)

        h = hk.Conv2D(self.out_channels, kernel_shape=3, stride=1, padding="SAME")(h)
        h = hk.BatchNorm(True, True, 0.9)(h, is_training)

        # --- Shortcut Path ---
        shortcut = hk.Conv2D(
            self.out_channels, kernel_shape=1, stride=self.stride, padding="SAME"
        )(x)

        return jax.nn.leaky_relu(h + shortcut, negative_slope=0.2)


class Generator(hk.Module):
    def __call__(self, z, is_training: bool):
        # 1. Initial Projection: Project to 4x4 with 512 channels
        x = hk.Linear(4 * 4 * 256)(z)
        x = jnp.reshape(x, [-1, 4, 4, 256])

        # 2. Upsampling Chain: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        # Channel strategy: Start wide (512), slowly narrow down to RGB (3)
        x = GenResBlock(128)(x, is_training)  # 8x8
        x = GenResBlock(64)(x, is_training)  # 16x16
        x = GenResBlock(32)(x, is_training)  # 32x32
        x = GenResBlock(32)(x, is_training)  # 64x64
        x = GenResBlock(32)(x, is_training)  # 128x128

        # Final layer: Get to 256x256 and 3 channels
        x = hk.Conv2DTranspose(3, kernel_shape=4, stride=2, padding="SAME")(x)
        return jax.nn.tanh(x)


class Discriminator(hk.Module):
    def __call__(self, x, is_training: bool):
        # Downsampling Chain: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        x = DisResBlock(32)(x, is_training)  # 128
        x = DisResBlock(32)(x, is_training)  # 64
        x = DisResBlock(32)(x, is_training)  # 32
        x = DisResBlock(64)(x, is_training)  # 16
        x = DisResBlock(128)(x, is_training)  # 8
        x = DisResBlock(256)(x, is_training)  # 4

        x = hk.Flatten()(x)
        return hk.Linear(1)(x)


def build_network() -> tuple[hk.TransformedWithState, hk.TransformedWithState]:
    def gen(z, is_training: bool = True):
        return Generator()(z, is_training)

    def dis(x, is_training: bool = True):
        return Discriminator()(x, is_training)

    return hk.transform_with_state(gen), hk.transform_with_state(dis)
