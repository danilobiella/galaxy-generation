import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models import (
    AttentionBlock,
    Downsample,
    ResnetBlock,
    SinusoidalPosEmb,
    UNet,
    Upsample,
    build_network,
    is_conditional,
)

IMAGE_SIZE = 8
NUM_CLASSES = 4
DIM = 8
DIM_MULTS = (1, 2)


def small_unet(num_classes=None, dim=DIM, dim_mults=DIM_MULTS, out_channels=3):
    """The real U-Net is 51M params; shrink it so tests stay fast.

    `dim` stays a multiple of 8 because every ResnetBlock and the final
    projection use GroupNorm with 8 groups.
    """

    @hk.transform
    def net(input):
        return UNet(
            dim=dim,
            dim_mults=dim_mults,
            out_channels=out_channels,
            num_classes=num_classes,
        )(input)

    return net


def unet_inputs(key, n=2, num_classes=None, size=IMAGE_SIZE):
    x = jax.random.normal(key, (n, size, size, 3))
    t = jnp.arange(1, n + 1, dtype=jnp.int32)
    labels = None if num_classes is None else jnp.arange(n) % num_classes
    return (x, t, labels)


def flat_names(params):
    return {f"{module}/{name}" for module, block in params.items() for name in block}


def dead_parameters(grads):
    return sorted(
        f"{module}/{name}"
        for module, block in grads.items()
        for name, g in block.items()
        if not np.any(np.asarray(g))
    )


class TestSinusoidalPosEmb:
    def transformed(self, dim):
        @hk.transform
        def f(t):
            return SinusoidalPosEmb(dim)(t)

        return f

    def test_output_is_one_row_per_timestep(self):
        f = self.transformed(16)
        t = jnp.arange(5)
        params = f.init(jax.random.PRNGKey(0), t)
        assert f.apply(params, None, t).shape == (5, 16)

    def test_has_no_parameters(self):
        f = self.transformed(16)
        assert f.init(jax.random.PRNGKey(0), jnp.arange(3)) == {}

    def test_first_half_is_sin_and_second_half_is_cos(self):
        f = self.transformed(16)
        t = jnp.zeros((1,))
        out = np.asarray(f.apply({}, None, t))
        np.testing.assert_allclose(out[0, :8], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[0, 8:], 1.0, atol=1e-6)

    def test_distinct_timesteps_get_distinct_embeddings(self):
        f = self.transformed(16)
        out = np.asarray(f.apply({}, None, jnp.arange(10)))
        assert len({tuple(np.round(row, 5)) for row in out}) == 10

    def test_matches_the_closed_form(self):
        dim = 16
        t = jnp.arange(4, dtype=jnp.float32)
        half = dim // 2
        freqs = np.exp(np.arange(half) * -(np.log(10000) / (half - 1)))
        angles = np.asarray(t)[:, None] * freqs[None, :]
        expected = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
        out = self.transformed(dim).apply({}, None, t)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-6)


class TestDownsampleUpsample:
    def test_downsample_halves_each_spatial_axis(self):
        @hk.transform
        def f(x):
            return Downsample(16)(x)

        x = jnp.ones((2, 8, 8, 4))
        params = f.init(jax.random.PRNGKey(0), x)
        assert f.apply(params, None, x).shape == (2, 4, 4, 16)

    def test_upsample_doubles_each_spatial_axis(self):
        @hk.transform
        def f(x):
            return Upsample(16)(x)

        x = jnp.ones((2, 4, 4, 8))
        params = f.init(jax.random.PRNGKey(0), x)
        assert f.apply(params, None, x).shape == (2, 8, 8, 16)


class TestResnetBlock:
    def transformed(self, dim, groups=8):
        @hk.transform
        def f(x, time_emb):
            return ResnetBlock(dim, groups=groups)(x, time_emb)

        return f

    def test_sets_channels_and_preserves_spatial_dims(self):
        f = self.transformed(16)
        x = jnp.ones((2, 8, 8, 8))
        emb = jnp.ones((2, 12))
        params = f.init(jax.random.PRNGKey(0), x, emb)
        assert f.apply(params, None, x, emb).shape == (2, 8, 8, 16)

    def test_accepts_input_whose_channels_already_match(self):
        f = self.transformed(8)
        x = jnp.ones((2, 8, 8, 8))
        emb = jnp.ones((2, 12))
        params = f.init(jax.random.PRNGKey(0), x, emb)
        assert f.apply(params, None, x, emb).shape == (2, 8, 8, 8)

    def test_time_embedding_changes_the_output(self):
        f = self.transformed(8)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 8))
        emb_a = jnp.zeros((2, 12))
        emb_b = jnp.ones((2, 12))
        params = f.init(jax.random.PRNGKey(1), x, emb_a)
        out_a = f.apply(params, None, x, emb_a)
        out_b = f.apply(params, None, x, emb_b)
        assert not np.allclose(np.asarray(out_a), np.asarray(out_b))

    def test_group_norm_requires_channels_divisible_by_groups(self):
        f = self.transformed(4, groups=8)
        x = jnp.ones((2, 8, 8, 4))
        emb = jnp.ones((2, 12))
        with pytest.raises(ValueError):
            f.init(jax.random.PRNGKey(0), x, emb)


class TestAttentionBlock:
    def transformed(self, dim, num_heads=4):
        @hk.transform
        def f(x):
            return AttentionBlock(dim, num_heads=num_heads)(x)

        return f

    def test_preserves_shape(self):
        f = self.transformed(8)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4, 8))
        params = f.init(jax.random.PRNGKey(1), x)
        assert f.apply(params, None, x).shape == x.shape

    def test_mixes_information_across_spatial_positions(self):
        f = self.transformed(8)
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 4, 8))
        params = f.init(jax.random.PRNGKey(1), x)

        bumped = x.at[0, 0, 0, :].add(10.0)
        out = np.asarray(f.apply(params, None, x))
        out_bumped = np.asarray(f.apply(params, None, bumped))

        assert not np.allclose(out[0, 3, 3], out_bumped[0, 3, 3])


class TestUNetShapes:
    def test_output_matches_the_input_image_shape(self):
        net = small_unet()
        inputs = unet_inputs(jax.random.PRNGKey(0))
        params = net.init(jax.random.PRNGKey(1), inputs)
        out = net.apply(params, None, inputs)
        assert out.shape == (2, IMAGE_SIZE, IMAGE_SIZE, 3)

    def test_out_channels_is_configurable(self):
        net = small_unet(out_channels=1)
        inputs = unet_inputs(jax.random.PRNGKey(0))
        params = net.init(jax.random.PRNGKey(1), inputs)
        assert net.apply(params, None, inputs).shape[-1] == 1

    def test_samples_in_a_batch_are_independent(self):
        net = small_unet()
        x, t, _ = unet_inputs(jax.random.PRNGKey(0), n=3)
        params = net.init(jax.random.PRNGKey(1), (x, t, None))

        batched = np.asarray(net.apply(params, None, (x, t, None)))
        alone = np.asarray(net.apply(params, None, (x[1:2], t[1:2], None)))

        np.testing.assert_allclose(batched[1], alone[0], rtol=1e-4, atol=1e-5)

    def test_is_deterministic_and_ignores_the_rng(self):
        net = small_unet()
        inputs = unet_inputs(jax.random.PRNGKey(0))
        params = net.init(jax.random.PRNGKey(1), inputs)
        a = net.apply(params, jax.random.PRNGKey(2), inputs)
        b = net.apply(params, jax.random.PRNGKey(99), inputs)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestConditioning:
    def test_unconditional_network_accepts_no_labels(self):
        net = small_unet()
        inputs = unet_inputs(jax.random.PRNGKey(0))
        params = net.init(jax.random.PRNGKey(1), inputs)
        assert net.apply(params, None, inputs).shape[0] == 2

    def test_conditional_network_rejects_missing_labels(self):
        net = small_unet(num_classes=NUM_CLASSES)
        x, t, _ = unet_inputs(jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="without labels"):
            net.init(jax.random.PRNGKey(1), (x, t, None))

    def test_different_labels_produce_different_outputs(self):
        net = small_unet(num_classes=NUM_CLASSES)
        x, t, _ = unet_inputs(jax.random.PRNGKey(0), n=2)
        params = net.init(jax.random.PRNGKey(1), (x, t, jnp.zeros((2,), jnp.int32)))

        out_a = net.apply(params, None, (x, t, jnp.zeros((2,), jnp.int32)))
        out_b = net.apply(params, None, (x, t, jnp.ones((2,), jnp.int32)))

        assert not np.allclose(np.asarray(out_a), np.asarray(out_b))

    def test_embedding_table_has_a_row_for_the_null_token(self):
        net = small_unet(num_classes=NUM_CLASSES)
        inputs = unet_inputs(jax.random.PRNGKey(0), num_classes=NUM_CLASSES)
        params = net.init(jax.random.PRNGKey(1), inputs)

        embeddings = next(
            block["embeddings"]
            for module, block in params.items()
            if "class_embed" in module
        )
        assert embeddings.shape[0] == NUM_CLASSES + 1

    def test_null_token_behaves_as_its_own_class(self):
        net = small_unet(num_classes=NUM_CLASSES)
        x, t, _ = unet_inputs(jax.random.PRNGKey(0), n=1)
        null = jnp.full((1,), NUM_CLASSES, dtype=jnp.int32)
        params = net.init(jax.random.PRNGKey(1), (x, t, null))

        null_out = np.asarray(net.apply(params, None, (x, t, null)))
        assert np.all(np.isfinite(null_out))

        for label in range(NUM_CLASSES):
            other = jnp.full((1,), label, dtype=jnp.int32)
            out = np.asarray(net.apply(params, None, (x, t, other)))
            assert not np.allclose(null_out, out)

    def test_conditioning_adds_only_the_embedding_table(self):
        inputs = unet_inputs(jax.random.PRNGKey(0))
        uncond = small_unet().init(jax.random.PRNGKey(1), inputs)

        cond_inputs = unet_inputs(jax.random.PRNGKey(0), num_classes=NUM_CLASSES)
        cond = small_unet(num_classes=NUM_CLASSES).init(
            jax.random.PRNGKey(1), cond_inputs
        )

        extra = flat_names(cond) - flat_names(uncond)
        assert flat_names(uncond) <= flat_names(cond)
        assert all("class_embed" in name for name in extra)


class TestIsConditional:
    def test_true_for_a_conditional_parameter_tree(self):
        net = small_unet(num_classes=NUM_CLASSES)
        inputs = unet_inputs(jax.random.PRNGKey(0), num_classes=NUM_CLASSES)
        assert is_conditional(net.init(jax.random.PRNGKey(1), inputs))

    def test_false_for_an_unconditional_parameter_tree(self):
        net = small_unet()
        inputs = unet_inputs(jax.random.PRNGKey(0))
        assert not is_conditional(net.init(jax.random.PRNGKey(1), inputs))

    def test_false_for_an_empty_tree(self):
        assert not is_conditional({})


class TestGradients:
    def test_gradients_reach_every_parameter(self):
        net = small_unet(num_classes=NUM_CLASSES)
        inputs = unet_inputs(jax.random.PRNGKey(0), num_classes=NUM_CLASSES)
        params = net.init(jax.random.PRNGKey(1), inputs)

        def loss(p):
            return jnp.sum(net.apply(p, None, inputs) ** 2)

        assert dead_parameters(jax.grad(loss)(params)) == []


class TestBuildNetwork:
    def test_returns_an_init_apply_pair(self):
        net = build_network()
        assert isinstance(net, hk.Transformed)
        assert callable(net.init) and callable(net.apply)
