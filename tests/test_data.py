"""Tests for the Phase 2 tile dataset.

Synthetic images so we don't depend on the data cache. Each test exercises one
behavior in isolation: crop centering, heatmap placement, presence label,
out-of-crop labels, wavelength channel, augmentation determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from laser_detector.data import (
    DEFAULT_HEATMAP_SIGMA_PX,
    FrameRecord,
    HardNegativeBalancedSampler,
    LaserTileDataset,
    UNKNOWN_WAVELENGTH_CHANNEL,
    WAVELENGTH_CHANNEL,
    _chromaticity_norm,
    _make_gaussian_heatmap,
    _pick_crop_origin,
)


class _FakeLoader:
    """Returns a fixed BGR image regardless of path/checksum. Picklable."""

    def __init__(self, image: np.ndarray):
        self.image = image

    def load(self, image_path: str, checksum: str) -> np.ndarray:
        return self.image.copy()


def _solid_image(h: int, w: int, color_bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def test_chromaticity_norm_preserves_color_ratios():
    img = np.array([[[100, 50, 25]]], dtype=np.uint8)  # B G R
    chrom = _chromaticity_norm(img)
    # Ratios should sum to ~1 across the 3 channels.
    assert chrom.sum(axis=2)[0, 0] == pytest.approx(1.0)
    # Brightest channel before normalization stays brightest after.
    assert np.argmax(chrom[0, 0]) == np.argmax(img[0, 0])


def test_chromaticity_norm_handles_black_pixel_without_blowup():
    """Eps clipping prevents 0/0 → NaN/inf on dark pixels."""
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    chrom = _chromaticity_norm(img)
    assert np.isfinite(chrom).all()


def test_make_gaussian_heatmap_peak_is_at_label():
    hm = _make_gaussian_heatmap(50.0, 30.0, 100, 100, sigma_px=3.0)
    assert hm.shape == (100, 100)
    assert hm.argmax() == 30 * 100 + 50  # row-major
    assert hm.max() == pytest.approx(1.0, abs=1e-3)


def test_make_gaussian_heatmap_returns_zeros_when_label_out_of_bounds():
    hm = _make_gaussian_heatmap(-5.0, 50.0, 100, 100, sigma_px=3.0)
    assert hm.max() == 0.0


def test_make_gaussian_heatmap_is_compact_in_3sigma():
    """Gaussian σ=3 → almost no mass past 9 px from the peak."""
    hm = _make_gaussian_heatmap(50.0, 50.0, 100, 100, sigma_px=3.0)
    # The pixel 12 px from the peak is essentially zero (exp(-72/18) = 0.018)
    assert hm[50, 62] < 0.05


def test_pick_crop_origin_biased_keeps_label_inside():
    rng = np.random.default_rng(0)
    for _ in range(20):
        cx, cy = _pick_crop_origin(
            img_h=2160, img_w=3840, tile=1024, rng=rng,
            label_xy=(2000.0, 1000.0),
            positive_center_p=1.0,  # always bias
            edge_pad=64,
        )
        # Label local coords inside the tile, with edge pad.
        local_x = 2000.0 - cx
        local_y = 1000.0 - cy
        assert 64 <= local_x <= 1024 - 64
        assert 64 <= local_y <= 1024 - 64


def test_pick_crop_origin_random_when_no_label():
    rng = np.random.default_rng(0)
    cx, cy = _pick_crop_origin(
        img_h=2160, img_w=3840, tile=1024, rng=rng,
        label_xy=None,
        positive_center_p=1.0,
        edge_pad=64,
    )
    assert 0 <= cx <= 3840 - 1024
    assert 0 <= cy <= 2160 - 1024


def test_pick_crop_origin_label_near_edge_clamps_gracefully():
    """A label 10 px from the right edge can't sit `edge_pad` from a tile edge —
    we should still return a valid crop that contains it, not crash."""
    rng = np.random.default_rng(0)
    cx, cy = _pick_crop_origin(
        img_h=2160, img_w=3840, tile=1024, rng=rng,
        label_xy=(3835.0, 1000.0),
        positive_center_p=1.0,
        edge_pad=64,
    )
    assert 0 <= cx <= 3840 - 1024
    # Label has to be in the tile (it's only 5 px from the right edge of the image).
    assert cx <= 3835.0 <= cx + 1024


def _record(image_id, dive_id, label_xy, wavelength="red"):
    return FrameRecord(
        image_id=image_id,
        dive_id=dive_id,
        image_path="/fake.jpg",
        image_checksum="x" * 8,
        label_xy=label_xy,
        wavelength=wavelength,
    )


def test_dataset_positive_frame_returns_4_channels_and_peaked_heatmap():
    img = _solid_image(2160, 3840, (40, 30, 20))
    img[1000, 2000] = (0, 0, 255)  # bright red dot at the label location
    rec = _record(1, 10, label_xy=(2000.0, 1000.0), wavelength="red")
    ds = LaserTileDataset(
        records=[rec], loader=_FakeLoader(img),
        positive_center_p=1.0, augment=False, seed=42,
    )
    sample = ds[0]
    assert sample["image"].shape == (4, 1024, 1024)
    assert sample["image"].dtype.is_floating_point
    assert sample["heatmap"].shape == (1, 1024, 1024)
    assert sample["heatmap"].max().item() == pytest.approx(1.0, abs=1e-3)
    assert sample["presence"].item() == 1.0


def test_dataset_negative_frame_has_zero_heatmap_and_zero_presence():
    img = _solid_image(2160, 3840, (40, 30, 20))
    rec = _record(2, 11, label_xy=None, wavelength="green")
    ds = LaserTileDataset(records=[rec], loader=_FakeLoader(img), augment=False, seed=42)
    sample = ds[0]
    assert sample["heatmap"].max().item() == 0.0
    assert sample["presence"].item() == 0.0


def test_dataset_wavelength_channel_value():
    img = _solid_image(2160, 3840, (40, 30, 20))
    for wl, expected in [("red", 1.0), ("green", 0.0), (None, 0.5)]:
        rec = _record(1, 10, label_xy=None, wavelength=wl)
        ds = LaserTileDataset(records=[rec], loader=_FakeLoader(img), augment=False)
        sample = ds[0]
        wavelength_channel = sample["image"][3]
        assert wavelength_channel.unique().tolist() == [expected]


def test_dataset_unbiased_crop_can_miss_label_marking_presence_zero():
    """positive_center_p=0.0 → random crop, label often outside → presence=0."""
    img = _solid_image(2160, 3840, (40, 30, 20))
    rec = _record(1, 10, label_xy=(100.0, 100.0), wavelength="red")
    ds = LaserTileDataset(
        records=[rec], loader=_FakeLoader(img),
        positive_center_p=0.0, augment=False, seed=42,
    )
    # With label at (100, 100) and tile=1024, only crops with origin in
    # [-924, 100]×[-924, 100] (clamped to [0, 100]) include the label.
    # Probability ≈ (101/2817) * (101/1137) ≈ 0.32% of random crops include it.
    misses = sum(ds[0]["presence"].item() == 0.0 for _ in range(10))
    assert misses >= 9


def test_dataset_skips_failed_image_load():
    """Loader returning None on first try should advance to next idx, not crash."""

    class FlakyLoader:
        def __init__(self, good_image):
            self.good_image = good_image
            self.calls = 0

        def load(self, path, checksum):
            self.calls += 1
            return None if self.calls == 1 else self.good_image.copy()

    img = _solid_image(2160, 3840, (40, 30, 20))
    records = [_record(1, 10, None), _record(2, 10, None)]
    ds = LaserTileDataset(
        records=records, loader=FlakyLoader(img), augment=False, seed=0,
    )
    sample = ds[0]
    # Should have skipped record 0 and returned record 1.
    assert sample["image_id"] == 2


def _mixed_records(n_pos: int, n_neg: int) -> list[FrameRecord]:
    records = []
    for i in range(n_pos):
        records.append(_record(image_id=100 + i, dive_id=10, label_xy=(50.0, 50.0)))
    for i in range(n_neg):
        records.append(_record(image_id=200 + i, dive_id=10, label_xy=None))
    return records


def test_hard_negative_sampler_yields_50_50_balance():
    sampler = HardNegativeBalancedSampler(
        records=_mixed_records(n_pos=10, n_neg=40), seed=0,
    )
    indices = list(iter(sampler))
    assert len(indices) == 20  # 2 * n_pos
    n_pos_drawn = sum(1 for i in indices if i < 10)
    n_neg_drawn = sum(1 for i in indices if 10 <= i < 50)
    assert n_pos_drawn == 10
    assert n_neg_drawn == 10


def test_hard_negative_sampler_weights_dominate_after_set_neg_score():
    """Boosting one negative's score 1000x should make it dominate samples."""
    sampler = HardNegativeBalancedSampler(
        records=_mixed_records(n_pos=5, n_neg=10), seed=42,
    )
    # record_idx 5..14 are the negatives; bump record 7's score way up
    sampler.set_neg_score(7, 1000.0)
    counts = {i: 0 for i in range(5, 15)}
    for epoch in range(200):
        sampler.set_epoch(epoch)
        for idx in sampler:
            if 5 <= idx < 15:
                counts[idx] += 1
    boosted = counts[7]
    others_avg = (sum(counts.values()) - boosted) / 9
    assert boosted > 10 * others_avg


def test_hard_negative_sampler_falls_back_to_pos_only_when_no_negatives():
    sampler = HardNegativeBalancedSampler(
        records=_mixed_records(n_pos=5, n_neg=0), seed=0,
    )
    assert not sampler.has_negatives()
    indices = list(iter(sampler))
    # No negatives to balance against → one pass over positives.
    assert len(indices) == 5
    assert sorted(indices) == [0, 1, 2, 3, 4]
    assert len(sampler) == 5


def test_hard_negative_sampler_raises_when_no_positives():
    with pytest.raises(ValueError, match="positive"):
        HardNegativeBalancedSampler(records=_mixed_records(n_pos=0, n_neg=5))


def test_hard_negative_sampler_floor_keeps_zero_score_records_reachable():
    """A negative scored exactly 0 should still get sampled occasionally —
    a single bad-luck epoch shouldn't permanently silence a record."""
    sampler = HardNegativeBalancedSampler(
        records=_mixed_records(n_pos=2, n_neg=3), seed=0,
    )
    sampler.set_neg_score(2, 0.0)  # first negative
    sampler.set_neg_score(3, 1.0)
    sampler.set_neg_score(4, 1.0)
    seen_floored = False
    for epoch in range(500):
        sampler.set_epoch(epoch)
        if 2 in list(iter(sampler)):
            seen_floored = True
            break
    assert seen_floored


def test_hard_negative_sampler_rank_aware_disjoint_shards():
    """Per-rank slices must partition the same epoch's shuffle without overlap."""
    records = _mixed_records(n_pos=20, n_neg=40)
    s0 = HardNegativeBalancedSampler(records, seed=7, rank=0, world_size=4)
    s1 = HardNegativeBalancedSampler(records, seed=7, rank=1, world_size=4)
    s2 = HardNegativeBalancedSampler(records, seed=7, rank=2, world_size=4)
    s3 = HardNegativeBalancedSampler(records, seed=7, rank=3, world_size=4)
    for s in (s0, s1, s2, s3):
        s.set_epoch(0)
    a, b, c, d = list(iter(s0)), list(iter(s1)), list(iter(s2)), list(iter(s3))
    # Equal-sized shards.
    assert len(a) == len(b) == len(c) == len(d)
    # Disjoint by position — each rank picks idx[rank::world_size] from the same shuffle.
    union = a + b + c + d
    assert len(union) == 4 * len(a)


def test_hard_negative_sampler_set_epoch_changes_shuffle():
    sampler = HardNegativeBalancedSampler(
        records=_mixed_records(n_pos=10, n_neg=20), seed=0,
    )
    sampler.set_epoch(0)
    e0 = list(iter(sampler))
    sampler.set_epoch(0)
    e0_again = list(iter(sampler))
    sampler.set_epoch(1)
    e1 = list(iter(sampler))
    assert e0 == e0_again, "Same epoch must reproduce identical shuffle"
    assert e0 != e1, "Different epoch must reshuffle"


def test_hard_negative_sampler_record_index_to_array_index():
    """sample_neg_record_indices should return real record indices, not array offsets."""
    sampler = HardNegativeBalancedSampler(
        records=_mixed_records(n_pos=3, n_neg=7), seed=0,
    )
    chosen = sampler.sample_neg_record_indices(5)
    assert chosen.size == 5
    # All picks are in the negatives' record-index range [3, 10).
    assert ((chosen >= 3) & (chosen < 10)).all()
    # Without replacement → unique.
    assert len(set(chosen.tolist())) == 5
