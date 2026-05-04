"""Smoke + correctness tests for the Phase 2 model + losses."""

from __future__ import annotations

import pytest
import torch

from laser_detector.model import LaserDetector, bce_heatmap_loss, focal_heatmap_loss


@pytest.fixture(scope="module")
def model():
    # No imagenet download in tests — random init is enough for shape checks.
    return LaserDetector(encoder_weights=None)


def test_forward_shape_at_1024(model):
    x = torch.randn(2, 4, 1024, 1024)
    out = model(x)
    assert out["heatmap_logits"].shape == (2, 1, 1024, 1024)
    assert out["presence_logits"].shape == (2,)


def test_forward_shape_at_smaller_tile(model):
    """Check the network handles smaller inputs (useful for overfit tests)."""
    x = torch.randn(2, 4, 256, 256)
    out = model(x)
    assert out["heatmap_logits"].shape == (2, 1, 256, 256)
    assert out["presence_logits"].shape == (2,)


def test_focal_loss_zero_when_pred_matches_target_well():
    """A near-perfect prediction should produce a small loss."""
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 16, 16] = 1.0  # one peak
    # logits → sigmoid ≈ target
    logits = torch.full((1, 1, 32, 32), -10.0)  # σ(-10) ≈ 4.5e-5
    logits[0, 0, 16, 16] = 10.0  # σ(10) ≈ 1.0
    loss = focal_heatmap_loss(logits, target)
    assert loss.item() < 0.01


def test_focal_loss_higher_when_pred_misses():
    """Predicting the wrong peak location should hurt more than predicting nothing."""
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 16, 16] = 1.0
    # Wrong peak placement
    logits_wrong = torch.full((1, 1, 32, 32), -5.0)
    logits_wrong[0, 0, 5, 5] = 10.0
    # Predict-nothing baseline
    logits_zero = torch.full((1, 1, 32, 32), -5.0)
    assert focal_heatmap_loss(logits_wrong, target) > focal_heatmap_loss(logits_zero, target)


def test_focal_loss_handles_negative_tile():
    """All-zero target → no NaNs, finite gradient w.r.t. logits."""
    target = torch.zeros(1, 1, 32, 32)
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss = focal_heatmap_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_backward_pass_does_not_explode(model):
    """Both heads should produce finite grads on the encoder."""
    x = torch.randn(2, 4, 256, 256, requires_grad=False)
    target_heatmap = torch.zeros(2, 1, 256, 256)
    target_heatmap[0, 0, 100, 100] = 1.0
    target_presence = torch.tensor([1.0, 0.0])

    out = model(x)
    loss_hm = focal_heatmap_loss(out["heatmap_logits"], target_heatmap)
    loss_pres = torch.nn.functional.binary_cross_entropy_with_logits(
        out["presence_logits"], target_presence
    )
    total = loss_hm + 0.5 * loss_pres
    total.backward()

    # Sanity: encoder first-conv has finite grads.
    first_conv = next(model.unet.encoder.parameters())
    assert first_conv.grad is not None
    assert torch.isfinite(first_conv.grad).all()


def test_in_channels_4_first_conv():
    """We pass a 4-channel input; first conv must accept that."""
    m = LaserDetector(encoder_weights=None, in_channels=4)
    first_conv = next(
        c for c in m.unet.encoder.modules() if isinstance(c, torch.nn.Conv2d)
    )
    assert first_conv.in_channels == 4


def test_bce_heatmap_loss_pos_weight_increases_peak_penalty():
    """Missing the peak should hurt much more under high pos_weight."""
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 16, 16] = 1.0
    # Predict ~0 everywhere — misses the peak.
    logits = torch.full((1, 1, 32, 32), -10.0)

    loss_pw_1 = bce_heatmap_loss(logits, target, pos_weight=1.0)
    loss_pw_1000 = bce_heatmap_loss(logits, target, pos_weight=1000.0)
    # With 1000x pos_weight, the single missed peak dominates the loss.
    assert loss_pw_1000 > 100 * loss_pw_1


def test_bce_heatmap_loss_low_when_pred_matches_target():
    """High prediction at the peak + low elsewhere → small BCE loss."""
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 16, 16] = 1.0
    logits = torch.full((1, 1, 32, 32), -10.0)
    logits[0, 0, 16, 16] = 10.0
    loss = bce_heatmap_loss(logits, target, pos_weight=1000.0)
    assert loss.item() < 0.01


def test_bce_heatmap_loss_finite_grad_on_negative_tile():
    """All-zero target (negative tile) shouldn't blow up gradients."""
    target = torch.zeros(1, 1, 32, 32)
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss = bce_heatmap_loss(logits, target, pos_weight=1000.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
