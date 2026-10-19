"""Fetch pinned files over HTTP(S), checking their SHA-256 before they land."""

import hashlib
import urllib.request
from pathlib import Path


def sha256_of(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url: str, path: str | Path, sha256: str) -> Path:
    """Download `url` to `path`. The file is written next to `path` first and
    only moved into place if its hash matches, so a failed or tampered download
    never leaves a file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    print(f"Downloading {url} to {path}...")
    try:
        urllib.request.urlretrieve(url, partial)
        digest = sha256_of(partial)
        if digest != sha256:
            raise ValueError(f"{url} has sha256 {digest}, expected {sha256}.")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path
