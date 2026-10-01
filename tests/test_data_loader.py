import itertools
import pickle

import lmdb
import numpy as np
import pytest

from data_loader import augment_features, dihedral_augment, make_data_loaders

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

    return build


def sample_id(image):
    return int(round((float(image[CENTRE, CENTRE, 0]) + 1.0) / 2.0 * 255.0))


def marker_corner(image):
    return np.unravel_index(int(np.argmax(image[:, :, 0])), image.shape[:2])


def run_epoch(dataset):
    images, labels = [], []
    for i in range(len(dataset)):
        batch_x, batch_y = dataset[i]
        images.extend(np.asarray(batch_x))
        labels.extend(np.asarray(batch_y))
    return np.stack(images), np.asarray(labels)


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
        images, _ = ds[0]
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
        images, _ = ds[0]
        assert np.asarray(images).shape == (4, 4, 4, 3)

    def test_drop_remainder_discards_the_partial_batch(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(25), batch_size=4, seed=0)
        assert len(ds) == 6
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
    """Targets for the reshuffle work; these fail against the current loader.

    `strict=True` means the suite turns red once they start passing, which is the
    signal to delete the markers.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="no .repeat(): every epoch replays the same shuffle permutation",
    )
    def test_consecutive_epochs_use_different_orders(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0)
        first, _ = run_epoch(ds)
        second, _ = run_epoch(ds)
        assert [sample_id(i) for i in first] != [sample_id(i) for i in second]

    @pytest.mark.xfail(
        strict=True,
        reason="random_map seeds off the element index, so an image's "
        "augmentation is fixed until .repeat() gives it new indices",
    )
    def test_an_image_is_augmented_differently_across_epochs(self, make_lmdb):
        ds = make_data_loaders(make_lmdb(24), batch_size=4, seed=0, augment=True)

        def corner_by_id(images):
            return {sample_id(img): marker_corner(img) for img in images}

        first, _ = run_epoch(ds)
        second, _ = run_epoch(ds)
        assert corner_by_id(first) != corner_by_id(second)
