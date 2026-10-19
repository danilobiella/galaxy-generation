import hashlib

import pytest

from src.download import download


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"galaxy weights")
    return path


def test_download_lands_a_file_with_the_pinned_hash(tmp_path, source):
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "out" / "weights.safetensors"

    download(source.as_uri(), target, sha256)

    assert target.read_bytes() == source.read_bytes()


def test_download_with_a_wrong_hash_leaves_nothing_behind(tmp_path, source):
    target = tmp_path / "weights.safetensors"

    with pytest.raises(ValueError, match="sha256"):
        download(source.as_uri(), target, "0" * 64)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.bin"]
