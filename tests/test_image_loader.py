"""Tests for the image loader and JPEG cache."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    CachingLinearImageLoader,
    LocalFilesystemImageLoader,
    NullImageLoader,
)


class _MemoryLoader:
    """Test loader: returns synthetic images keyed by checksum, counts calls."""

    def __init__(self, size: tuple[int, int] = (32, 32)) -> None:
        self.size = size
        self.calls: list[str] = []

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        self.calls.append(checksum)
        rng = np.random.default_rng(int(checksum, 16) % (2**32))
        return rng.integers(0, 256, size=(*self.size, 3), dtype=np.uint8)


def test_null_loader_returns_none():
    loader = NullImageLoader()
    assert loader.load("anything.orf", "abc") is None


def test_local_filesystem_loader_reads_jpeg(tmp_path: Path):
    image = (np.random.default_rng(0).integers(0, 256, (24, 24, 3))).astype(np.uint8)
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), image)
    loader = LocalFilesystemImageLoader(tmp_path)
    loaded = loader.load("frame.jpg", "abc")
    assert loaded is not None
    assert loaded.shape == (24, 24, 3)


def test_local_filesystem_loader_returns_none_for_missing(tmp_path: Path):
    loader = LocalFilesystemImageLoader(tmp_path)
    assert loader.load("does_not_exist.jpg", "abc") is None


def test_caching_loader_writes_then_reads_cache(tmp_path: Path):
    inner = _MemoryLoader()
    cache = CachingImageLoader(
        inner=inner, cache_dir=tmp_path / "cache", jpeg_quality=90
    )
    checksum = "abcdef1234567890"

    first = cache.load("ignored", checksum)
    assert first is not None
    assert len(inner.calls) == 1
    assert cache.cache_path(checksum).exists()

    # Second call is served from disk; inner is not invoked again.
    second = cache.load("ignored", checksum)
    assert second is not None
    assert len(inner.calls) == 1
    # JPEG round-trip is lossy at quality 90 but shapes match exactly.
    assert second.shape == first.shape


def test_caching_loader_uses_two_level_fanout(tmp_path: Path):
    inner = _MemoryLoader()
    cache = CachingImageLoader(inner=inner, cache_dir=tmp_path / "cache")
    checksum = "deadbeef00ff"
    cache.load("ignored", checksum)
    expected = tmp_path / "cache" / "de" / "ad" / f"{checksum}.jpg"
    assert expected.exists()


def test_caching_loader_propagates_none_from_inner(tmp_path: Path):
    class _NoneLoader:
        def load(self, image_path: str, checksum: str) -> np.ndarray | None:
            return None

    cache = CachingImageLoader(inner=_NoneLoader(), cache_dir=tmp_path / "cache")
    assert cache.load("missing", "deadbeef") is None
    # Nothing gets written for misses
    assert not any((tmp_path / "cache").rglob("*.jpg"))


def test_caching_loader_rejects_empty_checksum(tmp_path: Path):
    cache = CachingImageLoader(inner=_MemoryLoader(), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError):
        cache.load("frame.jpg", "")


class _Linear16BitLoader:
    """Test loader that returns synthetic uint16 BGR images keyed by checksum."""

    def __init__(self, size: tuple[int, int] = (32, 32)) -> None:
        self.size = size
        self.calls: list[str] = []

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        self.calls.append(checksum)
        rng = np.random.default_rng(int(checksum, 16) % (2**32))
        return rng.integers(0, 65536, size=(*self.size, 3), dtype=np.uint16)


def test_linear_caching_loader_round_trips_uint16(tmp_path: Path):
    inner = _Linear16BitLoader()
    cache = CachingLinearImageLoader(inner=inner, cache_dir=tmp_path / "cache")
    checksum = "abcdef1234567890"

    first = cache.load("ignored", checksum)
    assert first is not None
    assert first.dtype == np.uint16
    assert len(inner.calls) == 1
    assert cache.cache_path(checksum).exists()
    assert cache.cache_path(checksum).suffix == ".png"

    second = cache.load("ignored", checksum)
    assert second is not None
    assert second.dtype == np.uint16
    assert len(inner.calls) == 1
    np.testing.assert_array_equal(first, second)


def test_linear_caching_loader_uses_two_level_fanout(tmp_path: Path):
    cache = CachingLinearImageLoader(
        inner=_Linear16BitLoader(), cache_dir=tmp_path / "cache"
    )
    checksum = "feedbeef00ff"
    cache.load("ignored", checksum)
    expected = tmp_path / "cache" / "fe" / "ed" / f"{checksum}.png"
    assert expected.exists()


def test_linear_caching_loader_rejects_empty_checksum(tmp_path: Path):
    cache = CachingLinearImageLoader(
        inner=_Linear16BitLoader(), cache_dir=tmp_path / "cache"
    )
    with pytest.raises(ValueError):
        cache.load("frame.png", "")


def test_linear_caching_loader_invalid_compression(tmp_path: Path):
    with pytest.raises(ValueError):
        CachingLinearImageLoader(
            inner=_Linear16BitLoader(),
            cache_dir=tmp_path / "cache",
            png_compression=11,
        )
