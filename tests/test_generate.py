import jax
import jax.numpy as jnp
import numpy as np
import pytest

import src.generate as generate_module
from src.generate import Generator, load_generator, random_seed, to_uint8
from src.weights import save_params

IMAGE_SIZE = (8, 8)
TIMESTEPS = 20
NUM_CLASSES = 4


class StubNetwork:
    """A network-shaped object whose output depends on params and labels.

    `Generator` only ever calls `.apply`, so this keeps the tests away from the
    real 51M-parameter U-Net without weakening what they check.
    """

    def __init__(self):
        self.calls = 0

    def init(self, rng, input):
        return {"scale": jnp.array(0.1)}

    def apply(self, params, rng, input):
        self.calls += 1
        x, t, y = input
        bias = 0.0 if y is None else y.reshape(-1, 1, 1, 1).astype(x.dtype)
        return params["scale"] * x + 0.01 * bias


@pytest.fixture
def stub_net(monkeypatch):
    net = StubNetwork()
    monkeypatch.setattr(generate_module, "build_network", lambda num_classes=None: net)
    return net


@pytest.fixture
def uncond(stub_net):
    return Generator({"scale": jnp.array(0.1)}, image_size=IMAGE_SIZE, timesteps=TIMESTEPS)


@pytest.fixture
def cond(stub_net):
    return Generator(
        {"scale": jnp.array(0.1)},
        image_size=IMAGE_SIZE,
        timesteps=TIMESTEPS,
        num_classes=NUM_CLASSES,
    )


class TestToUint8:
    def test_maps_the_model_range_onto_the_byte_range(self):
        out = to_uint8(jnp.array([-1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(out, [0, 128, 255])

    def test_clips_out_of_range_values(self):
        out = to_uint8(jnp.array([-5.0, 5.0]))
        np.testing.assert_array_equal(out, [0, 255])

    def test_returns_an_unsigned_byte_array(self):
        assert to_uint8(jnp.zeros((2, 4, 4, 3))).dtype == np.uint8

    def test_preserves_shape(self):
        assert to_uint8(jnp.zeros((2, 4, 4, 3))).shape == (2, 4, 4, 3)


class TestRandomSeed:
    def test_is_within_the_jax_seed_range(self):
        assert 0 <= random_seed() < 2**31

    def test_varies_between_calls(self):
        assert len({random_seed() for _ in range(5)}) > 1


class TestResolveLabels:
    def test_cycles_through_classes_when_none_given(self, cond):
        labels = cond.resolve_labels(6, None)
        np.testing.assert_array_equal(np.asarray(labels), [0, 1, 2, 3, 0, 1])

    def test_broadcasts_a_scalar(self, cond):
        np.testing.assert_array_equal(np.asarray(cond.resolve_labels(3, 2)), [2, 2, 2])

    def test_accepts_an_explicit_sequence(self, cond):
        np.testing.assert_array_equal(np.asarray(cond.resolve_labels(3, [0, 3, 1])), [0, 3, 1])

    def test_rejects_a_wrong_length_sequence(self, cond):
        with pytest.raises(ValueError, match="Expected 3 labels"):
            cond.resolve_labels(3, [0, 1])

    def test_rejects_an_out_of_range_label(self, cond):
        with pytest.raises(ValueError, match=r"\[0, 3\]"):
            cond.resolve_labels(2, NUM_CLASSES)

    def test_rejects_a_negative_label(self, cond):
        with pytest.raises(ValueError, match=r"\[0, 3\]"):
            cond.resolve_labels(2, -1)

    def test_unconditional_model_returns_none(self, uncond):
        assert uncond.resolve_labels(4, None) is None

    def test_unconditional_model_rejects_labels(self, uncond):
        with pytest.raises(ValueError, match="unconditional"):
            uncond.resolve_labels(4, 0)

    def test_the_null_token_is_not_a_valid_label(self, cond):
        """`num_classes` is reserved for the unconditional branch inside CFG."""
        with pytest.raises(ValueError):
            cond.resolve_labels(1, NUM_CLASSES)


class TestStepSchedule:
    def test_ddpm_walks_every_timestep_backwards(self, uncond):
        pairs = uncond.step_schedule("ddpm", steps=5)
        assert len(pairs) == TIMESTEPS
        assert pairs[0] == (TIMESTEPS - 1, TIMESTEPS - 2)
        assert pairs[-1] == (0, -1)

    def test_ddim_uses_the_requested_number_of_steps(self, uncond):
        assert len(uncond.step_schedule("ddim", steps=5)) == 5

    def test_ddim_ends_at_the_terminal_marker(self, uncond):
        assert uncond.step_schedule("ddim", steps=5)[-1][1] == -1

    def test_ddim_timesteps_descend(self, uncond):
        ts = [t for t, _ in uncond.step_schedule("ddim", steps=8)]
        assert ts == sorted(ts, reverse=True)

    def test_rejects_an_unknown_sampler(self, uncond):
        with pytest.raises(ValueError, match="Unknown sampler"):
            uncond.step_schedule("euler", steps=5)


class TestGenerate:
    def test_returns_uint8_images_of_the_configured_size(self, uncond):
        out = uncond.generate(n_samples=2, seed=0, steps=3)
        assert out.shape == (2, *IMAGE_SIZE, 3)
        assert out.dtype == np.uint8

    def test_is_reproducible_for_a_fixed_seed(self, uncond):
        a = uncond.generate(n_samples=2, seed=7, steps=3)
        b = uncond.generate(n_samples=2, seed=7, steps=3)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_images(self, uncond):
        a = uncond.generate(n_samples=2, seed=1, steps=3)
        b = uncond.generate(n_samples=2, seed=2, steps=3)
        assert not np.array_equal(a, b)

    def test_ddpm_also_produces_images(self, uncond):
        out = uncond.generate(n_samples=1, seed=0, sampler="ddpm")
        assert out.shape == (1, *IMAGE_SIZE, 3)

    def test_labels_change_the_output(self, cond):
        a = cond.generate(n_samples=2, seed=5, steps=3, labels=0)
        b = cond.generate(n_samples=2, seed=5, steps=3, labels=3)
        assert not np.array_equal(a, b)

    def test_explicit_params_override_the_stored_ones(self, uncond):
        a = uncond.generate(n_samples=1, seed=3, steps=3)
        b = uncond.generate(n_samples=1, seed=3, steps=3, params={"scale": jnp.array(0.9)})
        assert not np.array_equal(a, b)

    def test_rejects_an_unknown_sampler(self, uncond):
        with pytest.raises(ValueError, match="Unknown sampler"):
            uncond.generate(n_samples=1, seed=0, sampler="euler")


class TestGenerateTrajectory:
    def test_returns_one_frame_per_step_by_default(self, uncond):
        frames = uncond.generate_trajectory(n_samples=1, seed=0, steps=4)
        assert len(frames) == 4

    def test_stride_thins_the_frames(self, uncond):
        """Steps 0, 3, 6 by stride, plus step 7 because the last is always kept."""
        frames = uncond.generate_trajectory(n_samples=1, seed=0, steps=8, stride=3)
        assert len(frames) == 4

    def test_always_keeps_the_final_frame(self, uncond):
        """Whatever the stride, the last state is the actual sample."""
        frames = uncond.generate_trajectory(n_samples=1, seed=0, steps=7, stride=4)
        direct = uncond.generate_trajectory(n_samples=1, seed=0, steps=7, stride=1)
        np.testing.assert_array_equal(frames[-1], direct[-1])

    def test_frames_are_images(self, uncond):
        frames = uncond.generate_trajectory(n_samples=2, seed=0, steps=3)
        assert all(f.shape == (2, *IMAGE_SIZE, 3) and f.dtype == np.uint8 for f in frames)


class TestFromWeights:
    def test_missing_file_names_the_path(self, tmp_path, stub_net):
        missing = tmp_path / "nope.safetensors"
        with pytest.raises(FileNotFoundError, match="nope.safetensors"):
            Generator.from_weights(missing)

    def test_detects_a_conditional_checkpoint(self, tmp_path, stub_net):
        path = tmp_path / "cond.safetensors"
        save_params({"class_embed": {"embedding": np.zeros((5, 4), np.float32)}}, path)
        gen = Generator.from_weights(path, image_size=IMAGE_SIZE, timesteps=TIMESTEPS)
        assert gen.conditional

    def test_detects_an_unconditional_checkpoint(self, tmp_path, stub_net):
        path = tmp_path / "uncond.safetensors"
        save_params({"Conv_0": {"kernel": np.zeros((3, 3, 3, 4), np.float32)}}, path)
        gen = Generator.from_weights(path, image_size=IMAGE_SIZE, timesteps=TIMESTEPS)
        assert not gen.conditional

    def test_explicit_num_classes_wins(self, tmp_path, stub_net):
        path = tmp_path / "uncond.safetensors"
        save_params({"Conv_0": {"kernel": np.zeros((3, 3, 3, 4), np.float32)}}, path)
        gen = Generator.from_weights(
            path, image_size=IMAGE_SIZE, timesteps=TIMESTEPS, num_classes=7
        )
        assert gen.num_classes == 7


class TestLoadGenerator:
    def test_repeated_loads_return_the_same_object(self, tmp_path, stub_net):
        path = tmp_path / "w.safetensors"
        save_params({"Conv_0": {"kernel": np.zeros((3, 3, 3, 4), np.float32)}}, path)
        a = load_generator(path, image_size=IMAGE_SIZE, timesteps=TIMESTEPS)
        b = load_generator(path, image_size=IMAGE_SIZE, timesteps=TIMESTEPS)
        assert a is b
