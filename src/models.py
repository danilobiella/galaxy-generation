import haiku as hk
import jax
import jax.numpy as jnp


class Generator(hk.Module):
    def __call__(self, z, is_training: bool):
        """z: Latent vector [batch, 100]"""
        # 1. Project and reshape to 4x4
        x = hk.Linear(4 * 4 * 256, w_init=hk.initializers.RandomNormal(0.02))(z)
        x = jnp.reshape(x, [-1, 4, 4, 256])

        # 2. Upsampling blocks: 4->8->16->32->64->128->256
        configs = [(128, 2), (64, 2), (32, 2), (16, 2), (8, 2), (3, 2)]
        for i, (channels, stride) in enumerate(configs):
            is_last = i == len(configs) - 1
            x = hk.Conv2DTranspose(
                output_channels=channels,
                kernel_shape=4,
                stride=stride,
                padding="SAME",
                w_init=hk.initializers.RandomNormal(0.02),
            )(x)

            if not is_last:
                x = hk.BatchNorm(True, True, 0.9)(x, is_training)
                x = jax.nn.relu(x)
            else:
                x = jax.nn.tanh(x)  # Output pixels in [-1, 1]
        return x


class Discriminator(hk.Module):
    def __call__(self, x, is_training: bool):
        for channels in [8, 16, 32, 64, 128, 256]:
            x = hk.Conv2D(channels, kernel_shape=4, stride=2, padding="SAME")(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)

        x = hk.Flatten()(x)
        logits = hk.Linear(1)(x)
        return logits


def build_network() -> tuple[hk.TransformedWithState, hk.TransformedWithState]:
    def gen(z, is_training: bool = True):
        return Generator()(z, is_training)

    def dis(x, is_training: bool = True):
        return Discriminator()(x, is_training)

    gen_fn = hk.transform_with_state(gen)
    dis_fn = hk.transform_with_state(dis)

    return gen_fn, dis_fn
