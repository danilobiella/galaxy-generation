import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    IMAGE_SHAPE,
    NUM_CLASSES,
    T,
    constant_unet,
    label_encoding_unet,
)

from src.diffusion import (
    SAMPLERS,
    DiffusionSchedule,
    check_sampler,
    ddim_timesteps,
    make_samplers,
)


def noise_image(schedule, t, x_0, eps):
    a_hat = schedule.alpha_hat[t]
    return jnp.sqrt(a_hat) * x_0 + jnp.sqrt(1.0 - a_hat) * eps


def read_eps(samplers, schedule, t, labels, guidance, n_samples):
    """Recovers the network's guided prediction from a final (t_prev=-1) DDIM step."""
    x = jnp.zeros((n_samples, *IMAGE_SHAPE))
    out = samplers.ddim_step(
        {}, x, t, -1, jax.random.key(0), False, 0.0, labels, guidance
    )
    a_hat = schedule.alpha_hat[t]
    return -out * jnp.sqrt(a_hat) / jnp.sqrt(1.0 - a_hat)


class TestSchedule:
    def test_linear_beta_hits_both_endpoints(self):
        s = DiffusionSchedule.create(T=1000, beta_start=1e-4, beta_end=0.02)
        assert s.beta.shape == (1000,)
        assert float(s.beta[0]) == pytest.approx(1e-4)
        assert float(s.beta[-1]) == pytest.approx(0.02)

    def test_alpha_is_one_minus_beta(self, schedule):
        np.testing.assert_allclose(schedule.alpha, 1.0 - schedule.beta, rtol=1e-6)

    def test_alpha_hat_is_running_product_of_alpha(self, schedule):
        np.testing.assert_allclose(
            schedule.alpha_hat, np.cumprod(np.asarray(schedule.alpha)), rtol=1e-5
        )

    def test_alpha_hat_decays_monotonically_within_the_unit_interval(self):
        s = DiffusionSchedule.create(T=1000)
        a_hat = np.asarray(s.alpha_hat)
        assert np.all(np.diff(a_hat) < 0.0)
        assert 0.0 < a_hat[-1] < a_hat[0] < 1.0

    def test_T_matches_array_length(self, schedule):
        assert schedule.T == T == schedule.beta.shape[0]


class TestCheckSampler:
    @pytest.mark.parametrize("name", SAMPLERS)
    def test_accepts_and_returns_known_samplers(self, name):
        assert check_sampler(name) == name

    @pytest.mark.parametrize("name", ["DDIM", "ddpm ", "euler", ""])
    def test_rejects_anything_else(self, name):
        with pytest.raises(ValueError, match="Unknown sampler"):
            check_sampler(name)


class TestDdimTimesteps:
    def test_spans_the_full_range_in_descending_order(self):
        ts = ddim_timesteps(1000, 50)
        assert ts.shape == (50,)
        assert ts[0] == 999
        assert ts[-1] == 0
        assert np.all(np.diff(ts) < 0)

    @pytest.mark.parametrize("n_steps", [20, 50])
    def test_shipped_configurations_never_repeat_a_timestep(self, n_steps):
        ts = ddim_timesteps(1000, n_steps)
        assert len(np.unique(ts)) == n_steps

    def test_integer_dtype_survives_indexing(self):
        assert ddim_timesteps(1000, 20).dtype == np.int32

    def test_single_step_starts_at_the_last_timestep(self):
        np.testing.assert_array_equal(ddim_timesteps(1000, 1), [999])

    def test_full_length_reproduces_the_ddpm_trajectory(self):
        np.testing.assert_array_equal(ddim_timesteps(T, T), np.arange(T - 1, -1, -1))

    @pytest.mark.parametrize("n_steps", [0, -1, 1001])
    def test_rejects_out_of_range_step_counts(self, n_steps):
        with pytest.raises(ValueError, match="n_steps must be in"):
            ddim_timesteps(1000, n_steps)


class TestDdpmStep:
    def test_perfect_noise_prediction_recovers_x_0_exactly(self, schedule, params):
        eps = 0.3
        x_0 = jnp.linspace(-1.0, 1.0, 8 * 8 * 3).reshape(1, *IMAGE_SHAPE)
        samplers = make_samplers(constant_unet(eps), schedule, IMAGE_SHAPE)

        x_t = noise_image(schedule, 0, x_0, eps)
        out = samplers.ddpm_step(params, x_t, 0, jax.random.key(0))

        np.testing.assert_allclose(out, x_0, atol=1e-5)

    def test_final_step_adds_no_noise(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)
        x = jax.random.normal(jax.random.key(1), (2, *IMAGE_SHAPE))

        a = samplers.ddpm_step(params, x, 0, jax.random.key(0))
        b = samplers.ddpm_step(params, x, 0, jax.random.key(99))

        np.testing.assert_array_equal(a, b)

    def test_earlier_steps_do_add_noise(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)
        x = jax.random.normal(jax.random.key(1), (2, *IMAGE_SHAPE))

        a = samplers.ddpm_step(params, x, 5, jax.random.key(0))
        b = samplers.ddpm_step(params, x, 5, jax.random.key(99))

        assert not np.allclose(a, b)


class TestDdimStep:
    def test_perfect_noise_prediction_stays_on_the_same_trajectory(
        self, schedule, params
    ):
        eps = 0.3
        x_0 = jnp.linspace(-1.0, 1.0, 8 * 8 * 3).reshape(1, *IMAGE_SHAPE)
        samplers = make_samplers(constant_unet(eps), schedule, IMAGE_SHAPE)

        x_t = noise_image(schedule, 15, x_0, eps)
        out = samplers.ddim_step(
            params, x_t, 15, 10, jax.random.key(0), False, 0.0
        )

        np.testing.assert_allclose(
            out, noise_image(schedule, 10, x_0, eps), atol=1e-5
        )

    def test_eta_zero_is_deterministic(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)
        x = jax.random.normal(jax.random.key(1), (2, *IMAGE_SHAPE))

        a = samplers.ddim_step(params, x, 15, 10, jax.random.key(0), False, 0.0)
        b = samplers.ddim_step(params, x, 15, 10, jax.random.key(99), False, 0.0)

        np.testing.assert_array_equal(a, b)

    def test_positive_eta_injects_noise(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)
        x = jax.random.normal(jax.random.key(1), (2, *IMAGE_SHAPE))

        a = samplers.ddim_step(params, x, 15, 10, jax.random.key(0), False, 1.0)
        b = samplers.ddim_step(params, x, 15, 10, jax.random.key(99), False, 1.0)

        assert not np.allclose(a, b)

    def test_eta_one_matches_the_ddpm_posterior_variance(self, schedule, params):
        samplers = make_samplers(constant_unet(0.0), schedule, IMAGE_SHAPE)
        t, t_prev = 15, 14
        x = jnp.zeros((1, *IMAGE_SHAPE))

        out = samplers.ddim_step(
            params, x, t, t_prev, jax.random.key(0), False, 1.0
        )
        z = jax.random.normal(jax.random.key(0), x.shape)

        a_hat_t = schedule.alpha_hat[t]
        a_hat_prev = schedule.alpha_hat[t_prev]
        beta_tilde = (1.0 - a_hat_prev) / (1.0 - a_hat_t) * (1.0 - a_hat_t / a_hat_prev)

        np.testing.assert_allclose(out, jnp.sqrt(beta_tilde) * z, atol=1e-6)

    def test_eta_one_reproduces_the_ddpm_posterior_mean(self, schedule, params):
        samplers = make_samplers(constant_unet(0.4), schedule, IMAGE_SHAPE)
        t = 15
        x = jax.random.normal(jax.random.key(3), (2, *IMAGE_SHAPE))
        key = jax.random.key(0)
        z = jax.random.normal(key, x.shape)

        a_hat_t = schedule.alpha_hat[t]
        a_hat_prev = schedule.alpha_hat[t - 1]
        beta_tilde = (1.0 - a_hat_prev) / (1.0 - a_hat_t) * schedule.beta[t]

        ddim_mean = (
            samplers.ddim_step(params, x, t, t - 1, key, False, 1.0)
            - jnp.sqrt(beta_tilde) * z
        )
        ddpm_mean = samplers.ddpm_step(params, x, t, key) - jnp.sqrt(schedule.beta[t]) * z

        np.testing.assert_allclose(ddim_mean, ddpm_mean, atol=1e-5)

    def test_stochastic_step_preserves_the_marginal_variance(self, schedule, params):
        t, t_prev = 15, 10
        shape = (256, *IMAGE_SHAPE)
        eps = jax.random.normal(jax.random.key(11), shape)
        samplers = make_samplers(constant_unet(eps), schedule, IMAGE_SHAPE)

        x_t = jnp.sqrt(1.0 - schedule.alpha_hat[t]) * eps
        out = samplers.ddim_step(
            params, x_t, t, t_prev, jax.random.key(0), False, 1.0
        )

        expected = 1.0 - schedule.alpha_hat[t_prev]
        assert float(out.var()) == pytest.approx(float(expected), rel=0.05)

    def test_terminal_step_returns_the_x_0_prediction(self, schedule, params):
        eps = 0.3
        x_0 = jnp.linspace(-1.0, 1.0, 8 * 8 * 3).reshape(1, *IMAGE_SHAPE)
        samplers = make_samplers(constant_unet(eps), schedule, IMAGE_SHAPE)

        x_t = noise_image(schedule, T - 1, x_0, eps)
        out = samplers.ddim_step(
            params, x_t, T - 1, -1, jax.random.key(0), False, 0.0
        )

        np.testing.assert_allclose(out, x_0, atol=1e-4)

    def test_terminal_step_is_noiseless_even_at_eta_one(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)
        x = jax.random.normal(jax.random.key(1), (2, *IMAGE_SHAPE))

        a = samplers.ddim_step(params, x, T - 1, -1, jax.random.key(0), False, 1.0)
        b = samplers.ddim_step(params, x, T - 1, -1, jax.random.key(99), False, 1.0)

        np.testing.assert_array_equal(a, b)

    def test_clip_bounds_the_x_0_prediction(self, schedule, params):
        samplers = make_samplers(constant_unet(5.0), schedule, IMAGE_SHAPE)
        x = jnp.zeros((1, *IMAGE_SHAPE))

        unclipped = samplers.ddim_step(
            params, x, T - 1, -1, jax.random.key(0), False, 0.0
        )
        clipped = samplers.ddim_step(
            params, x, T - 1, -1, jax.random.key(0), True, 0.0
        )

        assert np.abs(unclipped).max() > 1.0
        assert np.abs(clipped).max() <= 1.0 + 1e-6


class TestClassifierFreeGuidance:
    @pytest.fixture
    def samplers(self, schedule):
        return make_samplers(
            label_encoding_unet(), schedule, IMAGE_SHAPE, num_classes=NUM_CLASSES
        )

    def test_guidance_one_is_the_plain_conditional_prediction(
        self, samplers, schedule
    ):
        labels = jnp.array([1, 3])
        eps = read_eps(samplers, schedule, 10, labels, 1.0, 2)

        np.testing.assert_allclose(eps[0], jnp.full(IMAGE_SHAPE, 1.0), atol=1e-4)
        np.testing.assert_allclose(eps[1], jnp.full(IMAGE_SHAPE, 3.0), atol=1e-4)

    def test_guidance_zero_falls_back_to_the_null_token(self, samplers, schedule):
        labels = jnp.array([1, 3])
        eps = read_eps(samplers, schedule, 10, labels, 0.0, 2)

        np.testing.assert_allclose(
            eps, jnp.full((2, *IMAGE_SHAPE), float(NUM_CLASSES)), atol=1e-4
        )

    @pytest.mark.parametrize("guidance", [0.5, 2.0, 7.5])
    def test_extrapolates_along_the_conditional_direction(
        self, samplers, schedule, guidance
    ):
        labels = jnp.array([1])
        eps = read_eps(samplers, schedule, 10, labels, guidance, 1)

        expected = NUM_CLASSES + guidance * (1 - NUM_CLASSES)
        np.testing.assert_allclose(eps, jnp.full((1, *IMAGE_SHAPE), expected), atol=1e-3)

    def test_one_forward_pass_per_step_over_a_doubled_batch(self, schedule, params):
        unet = label_encoding_unet()
        samplers = make_samplers(
            unet, schedule, IMAGE_SHAPE, num_classes=NUM_CLASSES
        )
        x = jnp.zeros((3, *IMAGE_SHAPE))

        samplers.ddpm_step(params, x, 5, jax.random.key(0), jnp.array([0, 1, 2]), 2.0)

        assert len(unet.seen_shapes) == 1
        x_shape, t_shape, y_shape = unet.seen_shapes[0]
        assert x_shape == (6, *IMAGE_SHAPE)
        assert t_shape == (6,)
        assert y_shape == (6,)

    def test_unconditional_samplers_pass_no_labels(self, schedule, params):
        unet = constant_unet(0.1)
        samplers = make_samplers(unet, schedule, IMAGE_SHAPE)
        x = jnp.zeros((3, *IMAGE_SHAPE))

        samplers.ddpm_step(params, x, 5, jax.random.key(0))

        x_shape, t_shape, y_shape = unet.seen_shapes[0]
        assert x_shape == (3, *IMAGE_SHAPE)
        assert t_shape == (3,)
        assert y_shape is None


class TestRunLoops:
    def test_run_ddim_returns_one_image_per_sample(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)
        out = samplers.run_ddim(4, 5, True, params, jax.random.key(0), 0.0)
        assert out.shape == (4, *IMAGE_SHAPE)

    def test_run_ddpm_returns_one_image_per_sample(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)
        out = samplers.run_ddpm(4, params, jax.random.key(0))
        assert out.shape == (4, *IMAGE_SHAPE)

    def test_same_key_reproduces_the_same_batch(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)

        a = samplers.run_ddim(2, 5, True, params, jax.random.key(7), 0.0)
        b = samplers.run_ddim(2, 5, True, params, jax.random.key(7), 0.0)

        np.testing.assert_array_equal(a, b)

    def test_different_keys_give_different_batches(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)

        a = samplers.run_ddim(2, 5, True, params, jax.random.key(7), 0.0)
        b = samplers.run_ddim(2, 5, True, params, jax.random.key(8), 0.0)

        assert not np.allclose(a, b)

    def test_run_ddim_honours_the_requested_step_count(self, schedule, params):
        unet = constant_unet(0.1)
        samplers = make_samplers(unet, schedule, IMAGE_SHAPE)

        samplers.run_ddim(1, 5, True, params, jax.random.key(0), 0.0)

        assert len(unet.seen_shapes) == 1

    def test_sweeping_guidance_does_not_recompile(self, schedule, params):
        samplers = make_samplers(
            label_encoding_unet(), schedule, IMAGE_SHAPE, num_classes=NUM_CLASSES
        )
        labels = jnp.array([0, 1])

        for guidance in (1.0, 2.0, 4.0, 7.5):
            samplers.run_ddim(
                2, 5, True, params, jax.random.key(0), 0.0, labels, guidance
            )

        assert samplers.run_ddim._cache_size() == 1

    def test_changing_step_count_does_recompile(self, schedule, params):
        samplers = make_samplers(constant_unet(0.1), schedule, IMAGE_SHAPE)

        samplers.run_ddim(2, 5, True, params, jax.random.key(0), 0.0)
        samplers.run_ddim(2, 10, True, params, jax.random.key(0), 0.0)

        assert samplers.run_ddim._cache_size() == 2
