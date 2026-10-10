import math

import haiku as hk
import jax
import jax.numpy as jnp


class SinusoidalPosEmb(hk.Module):
    """Encodes the integer timestep t into a dense vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class Downsample(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        return hk.Conv2D(self.dim, kernel_shape=4, stride=2, padding="SAME")(x)


class Upsample(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        return hk.Conv2DTranspose(self.dim, kernel_shape=4, stride=2, padding="SAME")(x)


class ResnetBlock(hk.Module):
    """Standard ResNet block with timestep embedding injection."""

    def __init__(self, dim, groups=8):
        super().__init__()
        self.dim = dim
        self.groups = groups

    def __call__(self, x, time_emb):
        # 1. Feature processing
        h = hk.Conv2D(self.dim, kernel_shape=3, padding="SAME")(x)
        h = hk.GroupNorm(self.groups)(h)
        h = jax.nn.swish(h)

        # 2. Time injection (broadcast to spatial dimensions)
        time_hidden = hk.Linear(self.dim)(jax.nn.swish(time_emb))
        h = h + time_hidden[:, None, None, :]

        # 3. Output processing
        h = hk.Conv2D(self.dim, kernel_shape=3, padding="SAME")(h)
        h = hk.GroupNorm(self.groups)(h)
        h = jax.nn.swish(h)

        # 4. Skip connection matching
        if x.shape[-1] != self.dim:
            x = hk.Conv2D(self.dim, kernel_shape=1, padding="SAME")(x)

        return x + h


class AttentionBlock(hk.Module):
    """Standard scaled dot-product attention for the bottleneck."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

    def __call__(self, x):
        B, H, W, C = x.shape

        # Flatten spatial dimensions
        x_flat = jnp.reshape(x, (B, H * W, C))

        # We use Haiku's built-in MultiHeadAttention
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.dim // self.num_heads,
            w_init=hk.initializers.VarianceScaling(1.0),
        )

        # Self-attention
        out = attn(x_flat, x_flat, x_flat)

        # Reshape back to spatial
        out = jnp.reshape(out, (B, H, W, C))

        # Residual connection
        return x + out


class UNet(hk.Module):
    def __init__(
        self,
        dim: int = 64,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        out_channels: int = 3,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.out_channels = out_channels
        self.num_classes = num_classes

    def __call__(self, input):
        x, time, labels = input

        # 1. Timestep Embedding setup
        time_dim = self.dim * 4
        t_emb = SinusoidalPosEmb(self.dim)(time)
        t_emb = hk.Linear(time_dim)(t_emb)
        t_emb = jax.nn.swish(t_emb)
        t_emb = hk.Linear(time_dim)(t_emb)

        # 1b. Class conditioning, sharing the timestep embedding pathway.
        # Row `num_classes` is the null token used by the unconditional branch.
        if self.num_classes is not None:
            if labels is None:
                raise ValueError("Conditional UNet called without labels.")
            t_emb = t_emb + hk.Embed(
                vocab_size=self.num_classes + 1,
                embed_dim=time_dim,
                name="class_embed",
            )(labels)

        # 2. Initial Convolution
        x = hk.Conv2D(self.dim, kernel_shape=3, padding="SAME")(x)
        skips = [x]

        # 3. Downsampling Path (Encoder)
        dims = [self.dim * m for m in self.dim_mults]
        for i, current_dim in enumerate(dims):
            x = ResnetBlock(current_dim)(x, t_emb)
            skips.append(x)

            x = ResnetBlock(current_dim)(x, t_emb)
            skips.append(x)

            # Key fix: Do NOT push the downsampled output to skips.
            if i != len(dims) - 1:
                x = Downsample(current_dim)(x)

        # 4. Bottleneck
        mid_dim = dims[-1]
        x = ResnetBlock(mid_dim)(x, t_emb)
        x = AttentionBlock(mid_dim)(x)
        x = ResnetBlock(mid_dim)(x, t_emb)

        # 5. Upsampling Path (Decoder)
        for i, current_dim in reversed(list(enumerate(dims))):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResnetBlock(current_dim)(x, t_emb)

            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResnetBlock(current_dim)(x, t_emb)

            if i != 0:
                x = Upsample(current_dim)(x)

        # Final skip connection from the initial conv
        x = jnp.concatenate([x, skips.pop()], axis=-1)
        x = ResnetBlock(self.dim)(x, t_emb)

        # 6. Final Projection back to image channel count (predicting noise)
        x = hk.Conv2D(self.dim, kernel_shape=3, padding="SAME")(x)
        x = hk.GroupNorm(8)(x)
        x = jax.nn.swish(x)
        x = hk.Conv2D(self.out_channels, kernel_shape=1, padding="SAME")(x)

        return x


def build_network(
    num_classes: int | None = None,
    dim: int = 64,
    dim_mults: tuple[int, ...] = (1, 2, 4, 8),
):
    """Haiku-transformed U-Net taking `(images, timesteps, labels)`.

    `num_classes=None` builds the unconditional model, with exactly the same
    parameter names as before class conditioning existed, so old checkpoints
    still load. Pass `labels=None` when calling it.
    """

    @hk.transform
    def unet(input):
        model = UNet(
            dim=dim,
            dim_mults=dim_mults,
            out_channels=3,
            num_classes=num_classes,
        )
        return model(input)

    return unet


def is_conditional(params) -> bool:
    """Whether a parameter tree came from a class-conditional network."""
    return any("class_embed" in key for key in params)
