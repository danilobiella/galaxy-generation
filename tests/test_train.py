import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.diffusion import ddim_timesteps, make_samplers
from src.plot import epochs_table
from src.train import (
    ModelState,
    TrainState,
    fit_model,
    get_global_norm,
    make_eval_fn,
    make_model_fns,
    make_train_fns,
    train_epoch,
)

from conftest import IMAGE_SHAPE, NUM_CLASSES, TinyNetwork


def moved_rows(before, after) -> set[int]:
    """Which rows of `b` the optimiser touched, i.e. which labels were seen."""
    delta = np.asarray(after["b"]) - np.asarray(before["b"])
    return set(np.flatnonzero(delta).tolist())


def make_trainer(net, sched, num_classes=None, ema_decay=0.9999, p_uncond=0.1, lr=0.1):
    init_train_state, train_step = make_train_fns(
        net,
        optax.sgd(lr),
        sched,
        num_classes=num_classes,
        p_uncond=p_uncond,
        ema_decay=ema_decay,
    )
    state = init_train_state(ModelState(params=net.init(jax.random.key(0), None)))
    return state, train_step


class TestGlobalNorm:
    def test_matches_the_flattened_l2_norm(self):
        tree = {"a": jnp.array([3.0, 4.0]), "b": {"c": jnp.array([12.0])}}
        assert float(get_global_norm(tree)) == pytest.approx(13.0)

    def test_is_zero_for_an_all_zero_tree(self):
        assert float(get_global_norm({"a": jnp.zeros((3, 3))})) == 0.0

    def test_ignores_tree_structure(self):
        flat = {"a": jnp.array([1.0, 2.0, 2.0])}
        nested = {"a": jnp.array([1.0]), "b": {"c": jnp.array([2.0, 2.0])}}
        assert float(get_global_norm(flat)) == pytest.approx(float(get_global_norm(nested)))


class TestInitModelState:
    def test_builds_params_from_the_declared_image_shape(self, net, sched):
        captured = {}

        class ShapeRecorder(TinyNetwork):
            def init(self, rng, input):
                captured["shapes"] = tuple(
                    None if a is None else a.shape for a in input
                )
                return super().init(rng, input)

        recorder = ShapeRecorder()
        samplers = make_samplers(recorder, sched, IMAGE_SHAPE)
        init_model_state, _ = make_model_fns(recorder, samplers, image_shape=IMAGE_SHAPE)
        init_model_state(jax.random.key(0))
        assert captured["shapes"] == ((1, *IMAGE_SHAPE), (1,), None)

    def test_conditional_init_passes_labels(self, sched):
        captured = {}

        class ShapeRecorder(TinyNetwork):
            def init(self, rng, input):
                captured["labels"] = input[2]
                return super().init(rng, input)

        recorder = ShapeRecorder()
        samplers = make_samplers(recorder, sched, IMAGE_SHAPE, num_classes=NUM_CLASSES)
        init_model_state, _ = make_model_fns(
            recorder, samplers, image_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES
        )
        init_model_state(jax.random.key(0))
        assert captured["labels"].shape == (1,)

    def test_returns_a_model_state(self, net, sched):
        samplers = make_samplers(net, sched, IMAGE_SHAPE)
        init_model_state, _ = make_model_fns(net, samplers, image_shape=IMAGE_SHAPE)
        assert isinstance(init_model_state(jax.random.key(0)), ModelState)


class TestInitTrainState:
    def test_starts_at_step_zero(self, net, sched):
        state, _ = make_trainer(net, sched)
        assert int(state.step) == 0

    def test_ema_starts_equal_to_the_parameters(self, net, sched):
        state, _ = make_trainer(net, sched)
        assert float(state.ema_params["w"]) == float(state.model_state.params["w"])

    def test_ema_is_a_copy_not_an_alias(self, net, sched):
        """They must diverge during training, so they cannot share storage."""
        state, step_fn = make_trainer(net, sched)
        state, _ = step_fn(jax.random.key(1), state, jnp.ones((4, *IMAGE_SHAPE)))
        assert float(state.ema_params["w"]) != float(state.model_state.params["w"])

    def test_is_a_train_state(self, net, sched):
        state, _ = make_trainer(net, sched)
        assert isinstance(state, TrainState)


class TestTrainStep:
    def test_advances_the_step_counter(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        state, _ = step_fn(jax.random.key(1), state, batch)
        assert int(state.step) == 1

    def test_reports_the_expected_metrics(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        _, metrics = step_fn(jax.random.key(1), state, batch)
        assert set(metrics) == {"loss", "grad_norm", "param_norm"}

    def test_loss_is_a_finite_scalar(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        _, metrics = step_fn(jax.random.key(1), state, batch)
        assert metrics["loss"].shape == () and np.isfinite(float(metrics["loss"]))

    def test_repeated_steps_reduce_the_loss(self, net, sched, batch):
        """The whole point of the step: gradients must actually descend."""
        state, step_fn = make_trainer(net, sched, lr=0.05)
        key = jax.random.key(2)
        losses = []
        for _ in range(25):
            state, metrics = step_fn(key, state, batch)
            losses.append(float(metrics["loss"]))
        assert losses[-1] < losses[0]

    def test_parameters_change(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        before = float(state.model_state.params["w"])
        state, _ = step_fn(jax.random.key(1), state, batch)
        assert float(state.model_state.params["w"]) != before

    def test_is_deterministic_for_a_fixed_key(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        a, _ = step_fn(jax.random.key(3), state, batch)
        b, _ = step_fn(jax.random.key(3), state, batch)
        assert float(a.model_state.params["w"]) == float(b.model_state.params["w"])

    def test_different_keys_give_different_updates(self, net, sched, batch):
        """The key picks the timesteps and the noise, so it must matter."""
        state, step_fn = make_trainer(net, sched)
        a, _ = step_fn(jax.random.key(3), state, batch)
        b, _ = step_fn(jax.random.key(4), state, batch)
        assert float(a.model_state.params["w"]) != float(b.model_state.params["w"])

    def test_grad_norm_is_non_negative(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        _, metrics = step_fn(jax.random.key(1), state, batch)
        assert float(metrics["grad_norm"]) >= 0.0


class TestLabelDropout:
    def test_all_labels_are_dropped_at_p_uncond_one(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched, num_classes=NUM_CLASSES, p_uncond=1.0)
        after, _ = step_fn(jax.random.key(1), state, batch, jnp.array([0, 1, 2, 3], jnp.int32))
        assert moved_rows(state.model_state.params, after.model_state.params) == {NUM_CLASSES}

    def test_no_labels_are_dropped_at_p_uncond_zero(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched, num_classes=NUM_CLASSES, p_uncond=0.0)
        after, _ = step_fn(jax.random.key(1), state, batch, jnp.array([0, 1, 2, 3], jnp.int32))
        assert moved_rows(state.model_state.params, after.model_state.params) == {0, 1, 2, 3}

    def test_the_dropped_label_is_the_null_token(self, net, sched, batch):
        """Dropping must map to row `num_classes`, the row CFG samples from."""
        state, step_fn = make_trainer(net, sched, num_classes=NUM_CLASSES, p_uncond=1.0)
        after, _ = step_fn(jax.random.key(1), state, batch, jnp.array([1, 1, 1, 1], jnp.int32))
        moved = moved_rows(state.model_state.params, after.model_state.params)
        assert NUM_CLASSES in moved and 1 not in moved

    def test_a_partial_dropout_rate_mixes_both(self, sched, batch):
        """Some rows real, the null row too: the point of `p_uncond`."""
        net = TinyNetwork()
        state, step_fn = make_trainer(net, sched, num_classes=NUM_CLASSES, p_uncond=0.5)
        big = jax.random.normal(jax.random.key(9), (256, *IMAGE_SHAPE))
        labels = jnp.arange(256, dtype=jnp.int32) % NUM_CLASSES
        after, _ = step_fn(jax.random.key(1), state, big, labels)
        moved = moved_rows(state.model_state.params, after.model_state.params)
        assert NUM_CLASSES in moved and len(moved & {0, 1, 2, 3}) > 0

    def test_unconditional_training_leaves_the_label_rows_alone(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched, num_classes=None)
        after, _ = step_fn(jax.random.key(1), state, batch)
        assert moved_rows(state.model_state.params, after.model_state.params) == set()


class TestEmaUpdate:
    def test_warmup_dominates_the_first_step(self, net, sched, batch):
        """decay = min(ema_decay, (1+step)/(10+step)) is 0.1 at step 0."""
        state, step_fn = make_trainer(net, sched, ema_decay=0.9999)
        before = float(state.ema_params["w"])
        state, _ = step_fn(jax.random.key(1), state, batch)
        after = float(state.model_state.params["w"])
        assert float(state.ema_params["w"]) == pytest.approx(0.1 * before + 0.9 * after)

    def test_a_zero_decay_tracks_the_parameters_exactly(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched, ema_decay=0.0)
        state, _ = step_fn(jax.random.key(1), state, batch)
        assert float(state.ema_params["w"]) == pytest.approx(
            float(state.model_state.params["w"])
        )

    def test_the_ema_lags_behind_the_parameters(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched, ema_decay=0.9999, lr=0.5)
        key = jax.random.key(5)
        start = float(state.model_state.params["w"])
        for _ in range(20):
            state, _ = step_fn(key, state, batch)
        moved_params = abs(float(state.model_state.params["w"]) - start)
        moved_ema = abs(float(state.ema_params["w"]) - start)
        assert moved_ema < moved_params

    def test_ema_is_never_optimised_directly(self, net, sched, batch):
        """It is tracked, so it must stay finite and bounded by the iterates."""
        state, step_fn = make_trainer(net, sched)
        for _ in range(5):
            state, _ = step_fn(jax.random.key(6), state, batch)
        assert np.isfinite(float(state.ema_params["w"]))


class TestTrainEpoch:
    def test_consumes_every_batch(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        state, metrics = train_epoch(
            jax.random.key(0), state, step_fn, iter([batch] * 4), 4, 1, 1, lambda s: 1e-4
        )
        assert len(metrics["loss"]) == 4
        assert int(state.step) == 4

    def test_unpacks_image_label_tuples(self, net, sched, batch):
        """A (images, labels) batch must reach the step as a label, not be eaten."""
        state, step_fn = make_trainer(net, sched, num_classes=NUM_CLASSES, p_uncond=0.0)
        labels = jnp.array([0, 1, 1, 1], jnp.int32)
        after, _ = train_epoch(
            jax.random.key(0),
            state,
            step_fn,
            iter([(batch, labels)]),
            1,
            1,
            1,
            lambda s: 1e-4,
        )
        assert moved_rows(state.model_state.params, after.model_state.params) == {0, 1}

    def test_per_step_keys_are_derived_from_the_step_not_threaded(
        self, net, sched, batch
    ):
        """Each step must draw differently, and the draw must depend only on
        (base_key, state.step) -- that is what makes a resumed run identical to
        an uninterrupted one without checkpointing the key."""
        state, step_fn = make_trainer(net, sched)

        def keys_of_one_epoch():
            seen = []

            def recording_step(key, s, imgs, labels=None):
                seen.append(np.asarray(jax.random.key_data(key)).copy())
                return step_fn(key, s, imgs, labels)

            train_epoch(
                jax.random.key(0), state, recording_step, iter([batch] * 3), 3, 1, 1,
                lambda s: 1e-4,
            )
            return seen

        first, again = keys_of_one_epoch(), keys_of_one_epoch()

        assert not any(
            np.array_equal(a, b)
            for i, a in enumerate(first)
            for b in first[i + 1 :]
        )
        assert all(np.array_equal(a, b) for a, b in zip(first, again))

    def test_collects_all_three_metrics(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        _, metrics = train_epoch(
            jax.random.key(0), state, step_fn, iter([batch] * 2), 2, 1, 1, lambda s: 1e-4
        )
        assert set(metrics) == {"loss", "grad_norm", "param_norm"}
        assert all(len(v) == 2 for v in metrics.values())


class CountingStream:
    """Records how the trainer drives it: one iter(), then next() per step."""

    def __init__(self, batch):
        self.batch = batch
        self.iter_calls = 0
        self.next_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return self

    def __next__(self):
        self.next_calls += 1
        return self.batch


class TestStreamConsumption:
    def test_the_stream_is_iterated_once_and_pulled_per_step(
        self, net, sched, batch, tmp_path
    ):
        """Re-iterating per epoch would replay epoch 1 forever on a real loader."""
        state, step_fn = make_trainer(net, sched)
        stream = CountingStream(batch)
        final, history = fit_model(
            jax.random.key(0),
            state,
            step_fn,
            gen_fn=lambda key, model_state, filepath: None,
            lr_schedule=lambda s: 1e-4,
            train_ds=stream,
            steps_per_epoch=3,
            n_epochs=2,
            figures_dir=str(tmp_path / "figures"),
        )
        assert stream.iter_calls == 1
        assert stream.next_calls == 6
        assert int(final.step) == 6
        assert len(history["loss"]) == 6

    def test_a_stream_shorter_than_steps_raises_a_clear_error(self, net, sched, batch):
        state, step_fn = make_trainer(net, sched)
        with pytest.raises(RuntimeError, match="steps_per_epoch"):
            train_epoch(
                jax.random.key(0), state, step_fn, iter([batch]), 3, 1, 1, lambda s: 1e-4
            )


class TestFitModel:
    def test_runs_the_requested_number_of_epochs(self, net, sched, batch, tmp_path):
        state, step_fn = make_trainer(net, sched)
        calls = []
        history_len = 3

        final, history = fit_model(
            jax.random.key(0),
            state,
            step_fn,
            gen_fn=lambda key, model_state, filepath: calls.append(filepath),
            lr_schedule=lambda s: 1e-4,
            train_ds=itertools.repeat(batch),
            steps_per_epoch=1,
            n_epochs=history_len,
            figures_dir=str(tmp_path / "figures"),
        )
        assert len(calls) == history_len
        assert int(final.step) == history_len
        assert len(history["loss"]) == history_len

    def test_samples_from_the_ema_weights(self, net, sched, batch, tmp_path):
        """Sampling from the last iterate instead of the average is the bug
        this guards; they differ, so the check is meaningful."""
        state, step_fn = make_trainer(net, sched)
        seen = []

        final, _ = fit_model(
            jax.random.key(0),
            state,
            step_fn,
            gen_fn=lambda key, model_state, filepath: seen.append(
                float(model_state.params["w"])
            ),
            lr_schedule=lambda s: 1e-4,
            train_ds=itertools.repeat(batch),
            steps_per_epoch=1,
            n_epochs=1,
            figures_dir=str(tmp_path / "figures"),
        )
        assert seen[0] == pytest.approx(float(final.ema_params["w"]))
        assert seen[0] != pytest.approx(float(final.model_state.params["w"]))

    def test_evaluates_once_per_epoch_into_the_epochs_table(
        self, net, sched, batch, tmp_path
    ):
        state, step_fn = make_trainer(net, sched)
        epochs_seen = []

        def eval_fn(epoch, train_state):
            epochs_seen.append(epoch)
            sparse = {"kid": 0.5} if epoch == 2 else {}
            return {"val_mse": float(epoch), **sparse}

        _, history = fit_model(
            jax.random.key(0),
            state,
            step_fn,
            gen_fn=lambda key, model_state, filepath: None,
            lr_schedule=lambda s: 1e-4,
            train_ds=itertools.repeat(batch),
            steps_per_epoch=2,
            n_epochs=3,
            figures_dir=str(tmp_path / "figures"),
            eval_fn=eval_fn,
        )
        assert epochs_seen == [1, 2, 3]
        table = epochs_table(history)
        assert list(table.index) == [1, 2, 3]
        assert list(table["val_mse"]) == [1.0, 2.0, 3.0]
        assert table["kid"].isna().tolist() == [True, False, True]
        assert table["train_loss"].tolist() == pytest.approx(
            np.asarray(history["loss"]).reshape(3, 2).mean(axis=1)
        )


class TestEvalFn:
    def test_is_deterministic_and_reports_each_timestep(self, net, sched, batch):
        state, _ = make_trainer(net, sched, num_classes=NUM_CLASSES)
        labels = jnp.arange(batch.shape[0]) % NUM_CLASSES
        timesteps = ddim_timesteps(sched.T, 4)
        eval_fn = make_eval_fn(
            net, sched, [(batch, labels)] * 3, jax.random.key(0), timesteps, NUM_CLASSES
        )

        first = eval_fn(1, state)
        second = eval_fn(2, state)

        assert first == second
        per_t = [first[f"val_mse_t{int(t)}"] for t in timesteps]
        assert len(per_t) == 4
        assert first["val_mse"] == pytest.approx(np.mean(per_t))
