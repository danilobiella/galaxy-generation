import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.checkpoint import check_resumable, manager, restore, run_meta, save, saver
from src.train import ModelState, fit_model, make_train_fns

META = run_meta(
    steps_per_epoch=2,
    n_epochs=4,
    learning_rate=1e-4,
    warmup_steps=0,
    final_learning_rate=1e-6,
    data_seed=0,
    val_fraction=0.1,
    split_seed=0,
    val_split_hash="abc123",
)


class SeekableStream:
    """A list-backed stand-in for a grain iterator, which can be positioned.

    `fit_model` reads its place in the stream from the step count, so a resume
    needs a stream that implements the same tiny protocol grain's does.
    """

    def __init__(self, items):
        self.items = items
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = self.items[self.i]
        self.i += 1
        return item

    def get_state(self):
        return {"next_index": self.i}

    def set_state(self, state):
        self.i = state["next_index"]


def make_trainer(net, sched, lr=0.1):
    """Adam, unlike test_train.py's SGD: the optimiser state is what makes the
    restore target matter, and Adam's is a NamedTuple holding two moments."""
    optimizer = optax.adam(lr)
    init_train_state, train_step = make_train_fns(net, optimizer, sched, ema_decay=0.9)
    state = init_train_state(ModelState(params=net.init(jax.random.key(0), None)))
    return state, train_step, optimizer


@pytest.fixture
def mgr(tmp_path):
    return manager(tmp_path / "ckpt")


def saved_then_restored(mgr, net, sched, state, history):
    save(mgr, 1, state, history, META)
    mgr.wait_until_finished()
    fresh, _, _ = make_trainer(net, sched)
    return restore(mgr, fresh)


class TestRoundTrip:
    def test_restores_an_identical_state(self, net, sched, batch, mgr):
        state, step_fn, _ = make_trainer(net, sched)
        state, _ = step_fn(jax.random.key(1), state, batch)

        out = saved_then_restored(mgr, net, sched, state, {"loss": [0.5]})

        assert out is not None
        assert float(out.train_state.model_state.params["w"]) == pytest.approx(
            float(state.model_state.params["w"])
        )
        assert float(out.train_state.ema_params["w"]) == pytest.approx(
            float(state.ema_params["w"])
        )
        assert int(out.train_state.step) == int(state.step)
        assert out.history == {"loss": [0.5]}
        assert out.meta == META

    def test_the_optimiser_state_survives_as_optax_types(self, net, sched, batch, mgr):
        """Restoring without a target returns dicts instead of optax's own
        NamedTuples, and `update` then fails. This is the guard for that."""
        state, step_fn, optimizer = make_trainer(net, sched)
        state, _ = step_fn(jax.random.key(1), state, batch)

        out = saved_then_restored(mgr, net, sched, state, {})
        assert out is not None

        assert type(out.train_state.opt_state[0]).__name__ == "ScaleByAdamState"
        grads = jax.tree_util.tree_map(jnp.ones_like, out.train_state.model_state.params)
        updates, _ = optimizer.update(
            grads, out.train_state.opt_state, out.train_state.model_state.params
        )
        assert np.isfinite(float(updates["w"]))

    def test_no_checkpoint_restores_as_none(self, mgr):
        assert restore(mgr, None) is None


class TestRotation:
    """These never train, so they save a bare pytree rather than a TrainState."""

    @pytest.mark.parametrize(
        "keep, every, epochs, force_last, expected",
        [
            (2, 1, 5, False, [4, 5]),
            (10, 3, 9, False, [1, 3, 6, 9]),
            (10, 3, 5, True, [1, 3, 5]),
        ],
        ids=["max_to_keep_rotates", "interval_is_honoured", "force_writes_off_interval"],
    )
    def test_which_epochs_are_kept(
        self, tmp_path, keep, every, epochs, force_last, expected
    ):
        """orbax decides, so the training loop calls save every epoch. It writes
        on multiples of the interval plus the first epoch it sees, and `force`
        keeps the last epoch even when it falls off the interval."""
        mgr = manager(tmp_path / "ckpt", keep=keep, every=every)
        for epoch in range(1, epochs + 1):
            save(
                mgr,
                epoch,
                {"w": jnp.array(0.5)},
                {},
                META,
                force=force_last and epoch == epochs,
            )
        mgr.wait_until_finished()
        assert mgr.all_steps() == expected

    def test_saver_forces_on_the_last_epoch(self, tmp_path):
        """The callback `fit_model` drives, wired to the loop's own `is_last`."""
        mgr = manager(tmp_path / "ckpt", keep=10, every=3)
        save_checkpoint = saver(mgr, META)
        for epoch in range(1, 6):
            save_checkpoint(epoch, {"w": jnp.array(0.5)}, {}, epoch == 5)
        mgr.wait_until_finished()
        assert mgr.all_steps() == [1, 3, 5]


class TestResumeEquivalence:
    def test_an_interrupted_run_matches_an_uninterrupted_one(
        self, net, sched, batch, tmp_path
    ):
        """The whole point of the feature: 4 epochs straight through must equal
        2 epochs, a checkpoint, a restore, and 2 more.

        Every batch differs, so the resumed leg has to land on exactly the right
        position in the stream or the losses diverge.
        """
        steps, epochs = 2, 4
        stream = [
            (jax.random.normal(jax.random.key(i), batch.shape), None)
            for i in range(steps * epochs)
        ]
        common = dict(
            gen_fn=lambda key, model_state, filepath: None,
            lr_schedule=lambda s: 1e-4,
            steps_per_epoch=steps,
            figures_dir=str(tmp_path / "figures"),
        )

        state, step_fn, _ = make_trainer(net, sched)
        straight, full_history = fit_model(
            jax.random.key(0), state, step_fn, train_ds=SeekableStream(stream),
            n_epochs=epochs, **common,
        )

        mgr = manager(tmp_path / "ckpt", keep=2, every=1)
        state, step_fn, _ = make_trainer(net, sched)
        fit_model(
            jax.random.key(0), state, step_fn, train_ds=SeekableStream(stream),
            n_epochs=epochs // 2, history=None,
            checkpoint_fn=saver(mgr, META), **common,
        )
        mgr.wait_until_finished()

        fresh, step_fn, _ = make_trainer(net, sched)
        out = restore(mgr, fresh)
        assert out is not None
        assert int(out.train_state.step) == steps * epochs // 2

        resumed, resumed_history = fit_model(
            jax.random.key(0), out.train_state, step_fn,
            train_ds=SeekableStream(stream),
            n_epochs=epochs, history=out.history, **common,
        )

        assert int(resumed.step) == int(straight.step) == steps * epochs
        assert float(resumed.model_state.params["w"]) == pytest.approx(
            float(straight.model_state.params["w"])
        )
        assert float(resumed.ema_params["w"]) == pytest.approx(
            float(straight.ema_params["w"])
        )
        assert resumed_history["loss"] == pytest.approx(full_history["loss"])

    def test_resuming_a_stream_that_cannot_seek_raises(
        self, net, sched, batch, tmp_path
    ):
        """Silently replaying from the top would train the resumed epochs on the
        data the first half already saw."""
        state, step_fn, _ = make_trainer(net, sched)
        state = state._replace(step=jnp.array(4, dtype=jnp.int32))

        with pytest.raises(TypeError, match="cannot be positioned"):
            fit_model(
                jax.random.key(0), state, step_fn,
                gen_fn=lambda key, model_state, filepath: None,
                lr_schedule=lambda s: 1e-4,
                train_ds=iter([(batch, None)] * 8),
                steps_per_epoch=2,
                n_epochs=4,
                figures_dir=str(tmp_path / "figures"),
            )

    def test_a_restored_history_missing_a_metric_still_resumes(
        self, net, sched, batch, tmp_path
    ):
        """`{}` is what the rotation tests save, and an older checkpoint could
        carry a narrower history. It must not die after a full epoch."""
        state, step_fn, _ = make_trainer(net, sched)

        final, history = fit_model(
            jax.random.key(0), state, step_fn,
            gen_fn=lambda key, model_state, filepath: None,
            lr_schedule=lambda s: 1e-4,
            train_ds=SeekableStream([(batch, None)] * 2),
            steps_per_epoch=2,
            n_epochs=1,
            history={},
            figures_dir=str(tmp_path / "figures"),
        )
        assert set(history) == {"loss", "grad_norm", "param_norm", "epochs"}
        assert len(history["loss"]) == 2


class TestCheckResumable:
    def test_accepts_a_matching_run(self):
        check_resumable(META, dict(META))

    @pytest.mark.parametrize(
        "field, value",
        [
            ("steps_per_epoch", 3),
            ("n_epochs", 5),
            ("learning_rate", 2e-4),
            ("warmup_steps", 10),
            ("final_learning_rate", 1e-5),
            ("data_seed", 7),
            ("val_fraction", 0.2),
            ("split_seed", 1),
            ("val_split_hash", "def456"),
        ],
    )
    def test_rejects_any_changed_schedule_input(self, field, value):
        """None of these is in the checkpoint: the LR curve is replayed from the
        step count and the data stream is resumed by position."""
        with pytest.raises(ValueError, match="no longer matches"):
            check_resumable(META, {**META, field: value})
