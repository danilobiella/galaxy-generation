import itertools
import multiprocessing
import operator
import pickle
from concurrent.futures import ProcessPoolExecutor

import lmdb
import numpy as np
import pytest

from src.data_loader import (
    GalaxyLMDBDataSource,
    augment_features,
    close_envs,
    dihedral_augment,
    make_data_loaders,
    make_data_splits,
    split_hash,
    stratified_split,
)

SIZE = 8
CENTRE = SIZE // 2
NUM_CLASSES = 10


def an_image(size=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(size, size, 3)).astype(np.float32)


def all_dihedral_variants(image):
    """The 8 elements of the dihedral group, in the same order as memorization.py."""
    variants = []
    for flip in (False, True):
        base = image[:, ::-1] if flip else image
        for k in range(4):
            variants.append(np.ascontiguousarray(np.rot90(base, k=k, axes=(0, 1))))
    return variants


@pytest.fixture
def make_lmdb(tmp_path):
    """Builds a throwaway LMDB in the real on-disk format.

    Each image is a constant equal to its index with one bright corner pixel, so
    a sample can be identified after the pipeline (centre pixel) and the dihedral
    transform it received can be read off (corner position).
    """
    counter = itertools.count()

    def build(n_samples=24):
        path = tmp_path / f"db{next(counter)}"
        env = lmdb.open(str(path), map_size=2**24)
        with env.begin(write=True) as txn:
            for i in range(n_samples):
                img = np.full((SIZE, SIZE, 3), i, dtype=np.uint8)
                img[0, 0, :] = 255
                txn.put(
                    f"{i}".encode(),
                    pickle.dumps({"img": img, "label": i % NUM_CLASSES}),
                )
            txn.put(
                b"__metadata__",
                pickle.dumps(
                    {
                        "num_samples": n_samples,
                        "shape": (SIZE, SIZE, 3),
                        "dtype": "uint8",
                    }
                ),
            )
        env.close()
        return str(path)

    yield build
    close_envs()


def sample_id(image):
    return int(round((float(image[CENTRE, CENTRE, 0]) + 1.0) / 2.0 * 255.0))


def marker_corner(image):
    return np.unravel_index(int(np.argmax(image[:, :, 0])), image.shape[:2])


def run_epoch(data, epoch=0):
    """The batches making up one epoch; the stream is indexable, so epochs are too."""
    images, labels = [], []
    for i in range(data.steps_per_epoch):
        batch_x, batch_y = data.batches[epoch * data.steps_per_epoch + i]
        images.extend(np.asarray(batch_x))
        labels.extend(np.asarray(batch_y))
    return np.stack(images), np.asarray(labels)


def first_batch(data):
    return data.batches[0]


class TestDihedralAugment:
    def test_preserves_shape_and_dtype(self):
        image = an_image()
        out = dihedral_augment(image, np.random.default_rng(0))
        assert out.shape == image.shape
        assert out.dtype == image.dtype

    def test_output_is_contiguous(self):
        image = an_image()
        for seed in range(8):
            out = dihedral_augment(image, np.random.default_rng(seed))
            assert out.flags["C_CONTIGUOUS"]

    def test_output_is_always_one_of_the_eight_variants(self):
        image = an_image()
        variants = all_dihedral_variants(image)
        rng = np.random.default_rng(0)
        for _ in range(50):
            out = dihedral_augment(image, rng)
            assert any(np.array_equal(out, v) for v in variants)

    def test_reaches_every_one_of_the_eight_variants(self):
        image = an_image()
        rng = np.random.default_rng(0)
        seen = {dihedral_augment(image, rng).tobytes() for _ in range(200)}
        assert len(seen) == 8

    def test_is_deterministic_for_a_given_generator_seed(self):
        image = an_image()
        a = dihedral_augment(image, np.random.default_rng(7))
        b = dihedral_augment(image, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_does_not_mutate_its_input(self):
        image = an_image()
        before = image.copy()
        rng = np.random.default_rng(0)
        for _ in range(20):
            dihedral_augment(image, rng)
        np.testing.assert_array_equal(image, before)

    def test_identity_is_reachable(self):
        image = an_image()
        rng = np.random.default_rng(0)
        outs = [dihedral_augment(image, rng) for _ in range(200)]
        assert any(np.array_equal(out, image) for out in outs)

    def test_pixel_values_are_only_permuted(self):
        image = an_image()
        rng = np.random.default_rng(0)
        for _ in range(20):
            out = dihedral_augment(image, rng)
            np.testing.assert_allclose(
                np.sort(out, axis=None), np.sort(image, axis=None)
            )


class TestAugmentFeatures:
    def test_leaves_the_label_untouched(self):
        image = an_image()
        label = np.int32(7)
        _, out_label = augment_features((image, label), np.random.default_rng(0))
        assert out_label == label

    def test_augments_the_image(self):
        image = an_image()
        rng = np.random.default_rng(0)
        images = {
            augment_features((image, np.int32(0)), rng)[0].tobytes()
            for _ in range(200)
        }
        assert len(images) == 8


class TestPipeline:
    def test_batch_shape_and_dtype(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        images, _ = first_batch(ds)
        assert np.asarray(images).shape == (4, SIZE, SIZE, 3)
        assert np.asarray(images).dtype == np.float32

    def test_pixels_are_normalised_to_the_unit_interval(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        images, _ = run_epoch(ds)
        assert images.min() >= -1.0 and images.max() <= 1.0

    def test_resize_is_applied_when_requested(self, make_lmdb):
        ds = make_data_loaders(
            make_lmdb(24), batch_size=4, seed=0, image_size=(4, 4)
        )
        images, _ = first_batch(ds)
        assert np.asarray(images).shape == (4, 4, 4, 3)

    def test_drop_remainder_discards_the_partial_batch(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(25), batch_size=4, seed=0)
        assert ds.steps_per_epoch == 6
        images, _ = run_epoch(ds)
        assert len(images) == 24

    def test_one_epoch_visits_every_sample_exactly_once(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        images, _ = run_epoch(ds)
        assert sorted(sample_id(img) for img in images) == list(range(24))

    def test_labels_stay_attached_to_their_images(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        images, labels = run_epoch(ds)
        for image, label in zip(images, labels):
            assert label == sample_id(image) % NUM_CLASSES

    def test_shuffling_changes_the_order(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        images, _ = run_epoch(ds)
        assert [sample_id(img) for img in images] != list(range(24))

    def test_the_same_seed_reproduces_the_order(self, make_lmdb):
        a, _ = run_epoch(make_data_loaders(make_lmdb(24), batch_size=4, seed=0))
        b, _ = run_epoch(make_data_loaders(make_lmdb(24), batch_size=4, seed=0))
        assert [sample_id(i) for i in a] == [sample_id(i) for i in b]

    def test_different_seeds_give_different_orders(self, make_lmdb):
        a, _ = run_epoch(make_data_loaders(make_lmdb(24), batch_size=4, seed=0))
        b, _ = run_epoch(make_data_loaders(make_lmdb(24), batch_size=4, seed=1))
        assert [sample_id(i) for i in a] != [sample_id(i) for i in b]

    def test_without_augmentation_the_marker_stays_in_one_corner(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0, augment=False)
        images, _ = run_epoch(ds)
        assert {marker_corner(img) for img in images} == {(0, 0)}

    def test_augmentation_moves_the_marker_around_the_corners(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0, augment=True)
        images, _ = run_epoch(ds)
        corners = {marker_corner(img) for img in images}
        assert corners <= {(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)}
        assert len(corners) > 1


class TestPerEpochRandomness:

    def test_consecutive_epochs_use_different_orders(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        first, _ = run_epoch(ds, 0)
        second, _ = run_epoch(ds, 1)
        assert [sample_id(i) for i in first] != [sample_id(i) for i in second]

    def test_an_image_is_augmented_differently_across_epochs(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0, augment=True)

        def corner_by_id(images):
            return {sample_id(img): marker_corner(img) for img in images}

        first, _ = run_epoch(ds, 0)
        second, _ = run_epoch(ds, 1)
        assert corner_by_id(first) != corner_by_id(second)

    def test_every_epoch_is_an_exact_pass(self, make_lmdb):
        """Batching before repeating is what keeps this true."""
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        for epoch in range(6):
            images, _ = run_epoch(ds, epoch)
            assert sorted(sample_id(i) for i in images) == list(range(24))

    def test_every_epoch_is_an_exact_pass_with_a_remainder(self, make_lmdb):
        """25 records into batches of 4 drops one per epoch, never the same one."""
        ds = make_data_loaders(make_lmdb(25), batch_size=4, seed=0)
        dropped = set()
        for epoch in range(8):
            images, _ = run_epoch(ds, epoch)
            seen = [sample_id(i) for i in images]
            assert len(seen) == len(set(seen)) == 24
            dropped |= set(range(25)) - set(seen)
        assert len(dropped) > 1

    def test_every_epoch_is_ordered_differently(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        orders = [tuple(sample_id(i) for i in run_epoch(ds, e)[0]) for e in range(5)]
        assert len(set(orders)) == 5

    def test_labels_track_their_images_every_epoch(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        for epoch in range(3):
            images, labels = run_epoch(ds, epoch)
            for image, label in zip(images, labels):
                assert label == sample_id(image) % NUM_CLASSES


class TestResumePosition:
    """`fit_model` restores its place in the stream from `TrainState.step` alone,
    with `set_state({"next_index": step})`. These pin the properties that makes
    sound, so a grain upgrade that renamed the key or changed its meaning fails
    here rather than training silently on the wrong data.
    """

    def test_next_index_counts_batches_consumed(self, make_lmdb):
        data = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        batches = iter(data.batches)
        for consumed in range(5):
            assert batches.get_state()["next_index"] == consumed
            next(batches)

    @pytest.mark.parametrize("offset", [0, 5])
    def test_setting_next_index_resumes_the_identical_stream(self, make_lmdb, offset):
        """Offset 5 crosses an epoch boundary, since 24 samples at batch 4 is 6
        batches an epoch. Offset 0 must be a no-op, which is what lets the
        training loop seek unconditionally with no `if resuming` branch."""
        data = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        straight = [np.asarray(b[0]) for b in itertools.islice(iter(data.batches), 9)]

        batches = iter(data.batches)
        batches.set_state({"next_index": offset})
        resumed = [np.asarray(b[0]) for b in itertools.islice(batches, 4)]

        assert len(resumed) == 4
        assert all(np.array_equal(a, b) for a, b in zip(straight[offset:], resumed))


class TestDataSplits:
    def test_split_is_stratified_and_partitions_the_indices(self):
        labels = np.arange(100) % NUM_CLASSES
        train, val = stratified_split(labels, val_fraction=0.2, seed=0)
        assert set(train).isdisjoint(val)
        assert sorted(np.concatenate([train, val])) == list(range(100))
        assert np.bincount(labels[val], minlength=NUM_CLASSES).tolist() == [2] * NUM_CLASSES

    def test_val_batches_are_identical_and_unaugmented(self, make_lmdb):
        splits = make_data_splits(make_lmdb(100), batch_size=4, val_fraction=0.2)
        first = [np.asarray(x) for x, _ in splits.val]
        second = [np.asarray(x) for x, _ in splits.val]
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
        assert {marker_corner(img) for img in np.concatenate(first)} == {(0, 0)}

    def test_train_probe_is_fixed_train_data_matching_val_classes(self, make_lmdb):
        db = make_lmdb(100)
        splits = make_data_splits(db, batch_size=4, val_fraction=0.2)
        first = [(np.asarray(x), np.asarray(y)) for x, y in splits.train_probe]
        second = [np.asarray(x) for x, _ in splits.train_probe]
        assert all(np.array_equal(a, b) for (a, _), b in zip(first, second, strict=True))
        probe_images = np.concatenate([x for x, _ in first])
        probe_labels = np.concatenate([y for _, y in first])
        val_labels = np.concatenate([np.asarray(y) for _, y in splits.val])
        assert np.bincount(probe_labels, minlength=NUM_CLASSES).tolist() == (
            np.bincount(val_labels, minlength=NUM_CLASSES).tolist()
        )
        assert {sample_id(img) for img in probe_images} <= set(splits.train_idx.tolist())
        assert {marker_corner(img) for img in probe_images} == {(0, 0)}

    def test_train_probe_refuses_a_train_split_smaller_than_val(self, make_lmdb):
        with pytest.raises(ValueError):
            make_data_splits(make_lmdb(100), batch_size=4, val_fraction=0.6)

    def test_train_never_sees_val_samples(self, make_lmdb):
        splits = make_data_splits(make_lmdb(100), batch_size=4, val_fraction=0.2)
        val_ids = {sample_id(img) for x, _ in splits.val for img in np.asarray(x)}
        for epoch in range(3):
            images, _ = run_epoch(splits.train, epoch)
            assert val_ids.isdisjoint(sample_id(img) for img in images)

    def test_hash_tracks_the_split(self, make_lmdb):
        db = make_lmdb(100)
        a = make_data_splits(db, batch_size=4, val_fraction=0.2, split_seed=0)
        b = make_data_splits(db, batch_size=4, val_fraction=0.2, split_seed=0)
        c = make_data_splits(db, batch_size=4, val_fraction=0.2, split_seed=1)
        assert a.val_hash == b.val_hash != c.val_hash

    def test_split_is_pinned_across_library_versions(self):
        """Fails if a numpy upgrade changes the permutation stream."""
        _, val = stratified_split(np.arange(100) % NUM_CLASSES, 0.2, seed=0)
        assert split_hash(val) == "4d5ff5033266"

    @pytest.mark.parametrize("val_fraction", [0.1, 0.9])
    def test_a_class_missing_from_either_side_raises(self, val_fraction):
        labels = np.array([0] * 20 + [1] * 2)
        with pytest.raises(ValueError, match="Class 1"):
            stratified_split(labels, val_fraction=val_fraction, seed=0)

    @pytest.mark.parametrize("n, expected", [(5, 1), (15, 2), (25, 3)])
    def test_half_counts_round_up(self, n, expected):
        _, val = stratified_split(np.zeros(n, dtype=int), val_fraction=0.1, seed=0)
        assert len(val) == expected


class TestSharedEnv:
    def test_a_pickled_source_reads_in_a_spawned_process(self, make_lmdb):
        source = GalaxyLMDBDataSource(make_lmdb(24), np.arange(24))
        source[0]
        restored = pickle.loads(pickle.dumps(source))
        assert int(restored[3]["label"]) == 3

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            sample = pool.submit(operator.getitem, source, 3).result()
        assert int(sample["label"]) == 3

    def test_a_forked_process_refuses_the_inherited_handle(self, make_lmdb):
        source = GalaxyLMDBDataSource(make_lmdb(24), np.arange(24))
        source[0]

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            with pytest.raises(RuntimeError, match="spawn"):
                pool.submit(operator.getitem, source, 3).result()

    def test_closed_handles_are_reopened(self, make_lmdb):
        source = GalaxyLMDBDataSource(make_lmdb(24), np.arange(24))
        source[0]
        close_envs()
        assert int(source[3]["label"]) == 3
