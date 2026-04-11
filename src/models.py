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
        # --- Main Path ---
        h = hk.Conv2DTranspose(
            self.out_channels, kernel_shape=4, stride=self.stride, padding="SAME"
        )(x)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
        h = jax.nn.relu(h)

        h = hk.Conv2D(self.out_channels, kernel_shape=3, stride=1, padding="SAME")(h)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

        # --- Shortcut Path ---
        shortcut = hk.Conv2DTranspose(
            self.out_channels, kernel_shape=1, stride=self.stride, padding="SAME"
        )(x)

        return jax.nn.relu(h + shortcut)


class DisResBlock(hk.Module):
    """Downsampling Residual Block for the Discriminator (Critic)."""

    def __init__(self, out_channels, stride=2, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.stride = stride

    def __call__(self, x, is_training):
        # --- Main Path ---
        h = hk.Conv2D(
            self.out_channels, kernel_shape=4, stride=self.stride, padding="SAME"
        )(x)

        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
        h = jax.nn.leaky_relu(h, negative_slope=0.2)

        h = hk.Conv2D(self.out_channels, kernel_shape=3, stride=1, padding="SAME")(h)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

        # --- Shortcut Path ---
        shortcut = hk.Conv2D(
            self.out_channels, kernel_shape=1, stride=self.stride, padding="SAME"
        )(x)

        return jax.nn.leaky_relu(h + shortcut, negative_slope=0.2)


class Generator(hk.Module):
    def __call__(self, input, is_training: bool):
        z, c = input
        c_onehot = jax.nn.one_hot(c, 10)
        x = jnp.concatenate([z, c_onehot], axis=-1)

        x = hk.Linear(4 * 4 * 256)(x)
        x = jnp.reshape(x, [-1, 4, 4, 256])

        # upsampling chain: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        x = GenResBlock(256)(x, is_training)  # 8x8
        x = GenResBlock(256)(x, is_training)  # 16x16
        x = GenResBlock(128)(x, is_training)  # 32x32
        x = GenResBlock(128)(x, is_training)  # 64x64
        x = GenResBlock(64)(x, is_training)  # 128x128

        x = hk.Conv2DTranspose(3, kernel_shape=4, stride=2, padding="SAME")(x)

        return jax.nn.tanh(x)


class Discriminator(hk.Module):
    def __call__(self, input, is_training: bool):
        x, c = input

        x = DisResBlock(64)(x, is_training)
        x = DisResBlock(128)(x, is_training)
        x = DisResBlock(128)(x, is_training)
        x = DisResBlock(256)(x, is_training)  # spatial is now 16x16

        c_onehot = jax.nn.one_hot(c, 10)
        c_spatial = c_onehot[..., jnp.newaxis, jnp.newaxis, :]

        target_shape = x.shape[:-1] + (10,)
        c_spatial = jnp.broadcast_to(c_spatial, target_shape)

        x = jnp.concatenate([x, c_spatial], axis=-1)

        x = DisResBlock(256)(x, is_training)
        x = DisResBlock(256)(x, is_training)

        x = hk.Flatten()(x)
        return hk.Linear(1)(x)


def build_network() -> tuple[hk.Transformed, hk.Transformed]:
    def gen(z, is_training: bool = True):
        return Generator()(z, is_training)

    def dis(x, is_training: bool = True):
        return Discriminator()(x, is_training)

    return hk.transform(gen), hk.transform(dis)
