import haiku as hk
import jax
import jax.numpy as jnp


class SelfAttention(hk.Module):
    """Self-attention layer for coordinate-based global structure."""

    def __call__(self, x):
        batch, h, w, channels = x.shape
        num_pixels = h * w

        # Projections
        inter_channels = channels // 8
        f = hk.Conv2D(inter_channels, kernel_shape=1, with_bias=False, name="f")(x)
        g = hk.Conv2D(inter_channels, kernel_shape=1, with_bias=False, name="g")(x)
        h_val = hk.Conv2D(channels, kernel_shape=1, with_bias=False, name="h")(x)

        # Flatten for attention
        f = jnp.reshape(f, (batch, num_pixels, inter_channels))
        g = jnp.reshape(g, (batch, num_pixels, inter_channels))
        h_val = jnp.reshape(h_val, (batch, num_pixels, channels))

        # Attention Map
        s = jnp.matmul(f, jnp.transpose(g, (0, 2, 1)))
        beta = jax.nn.softmax(s, axis=-1)

        # Apply to Value
        o = jnp.matmul(beta, h_val)
        o = jnp.reshape(o, (batch, h, w, channels))

        # Gamma starts at 0.0 so the model learns to use attention slowly
        gamma = hk.get_parameter("gamma", [], init=jnp.zeros)
        return x + gamma * o


class GenResBlock(hk.Module):
    """Upsampling block using Resize-Convolution to avoid checkerboards."""

    def __init__(self, out_channels, upsample=True, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.upsample = upsample

    def __call__(self, x, is_training):
        # --- Main Path ---
        h = x
        if self.upsample:
            new_shape = [x.shape[1] * 2, x.shape[2] * 2]
            h = jax.image.resize(
                h, shape=([x.shape[0]] + new_shape + [x.shape[3]]), method="nearest"
            )

        h = hk.Conv2D(self.out_channels, kernel_shape=3, stride=1, padding="SAME")(h)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
        h = jax.nn.relu(h)

        h = hk.Conv2D(self.out_channels, kernel_shape=3, stride=1, padding="SAME")(h)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

        # --- Shortcut Path ---
        shortcut = x
        if self.upsample:
            new_shape = [x.shape[1] * 2, x.shape[2] * 2]
            shortcut = jax.image.resize(
                shortcut,
                shape=([x.shape[0]] + new_shape + [x.shape[3]]),
                method="nearest",
            )

        if shortcut.shape[-1] != self.out_channels:
            shortcut = hk.Conv2D(
                self.out_channels, kernel_shape=1, stride=1, padding="SAME"
            )(shortcut)

        return jax.nn.relu(h + shortcut)


class DisResBlock(hk.Module):
    """Standard Downsampling Residual Block."""

    def __init__(self, out_channels, stride=2, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.stride = stride

    def __call__(self, x, is_training):
        h = hk.Conv2D(
            self.out_channels, kernel_shape=3, stride=self.stride, padding="SAME"
        )(x)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
        h = jax.nn.leaky_relu(h, negative_slope=0.2)

        h = hk.Conv2D(self.out_channels, kernel_shape=3, stride=1, padding="SAME")(h)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

        shortcut = hk.Conv2D(
            self.out_channels, kernel_shape=1, stride=self.stride, padding="SAME"
        )(x)
        return jax.nn.leaky_relu(h + shortcut, negative_slope=0.2)


class Generator(hk.Module):
    def __call__(self, input, is_training: bool):
        z, c = input
        c_onehot = jax.nn.one_hot(c, 10)
        x = jnp.concatenate([z, c_onehot], axis=-1)

        # Start with a strong 512 base for complex concepts
        x = hk.Linear(4 * 4 * 512)(x)
        x = jnp.reshape(x, [-1, 4, 4, 512])

        x = GenResBlock(512)(x, is_training)  # 8x8
        x = GenResBlock(256)(x, is_training)  # 16x16
        x = GenResBlock(256)(x, is_training)  # 32x32

        x = SelfAttention()(x)

        x = GenResBlock(128)(x, is_training)  # 64x64
        x = GenResBlock(128)(x, is_training)  # 128x128
        x = GenResBlock(64, upsample=True)(x, is_training)  # 256x256

        x = hk.Conv2D(64, kernel_shape=3, stride=1, padding="SAME")(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)

        x = hk.Conv2D(3, kernel_shape=3, stride=1, padding="SAME")(x)
        return jax.nn.tanh(x)


class Discriminator(hk.Module):
    def __call__(self, input, is_training: bool):
        x, c = input

        # Label injection
        c_onehot = jax.nn.one_hot(c, 10)
        c_spatial = jnp.broadcast_to(c_onehot[:, None, None, :], x.shape[:-1] + (10,))
        x = jnp.concatenate([x, c_spatial], axis=-1)

        # Start with 64 filters to catch fine grain early
        x = DisResBlock(64)(x, is_training)  # 128x128
        x = DisResBlock(128)(x, is_training)  # 64x64
        x = DisResBlock(256)(x, is_training)  # 32x32

        x = SelfAttention()(x)

        x = DisResBlock(256)(x, is_training)  # 16x16
        x = DisResBlock(512)(x, is_training)  # 8x8
        x = DisResBlock(512)(x, is_training)  # 4x4

        x = hk.Flatten()(x)
        return hk.Linear(1)(x)


def build_network():
    def gen(z, is_training=True):
        return Generator()(z, is_training)

    def dis(x, is_training=True):
        return Discriminator()(x, is_training)

    return hk.transform(gen), hk.transform(dis)
