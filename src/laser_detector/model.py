"""Phase 2 supervised heatmap detector.

ResNet-34 U-Net (segmentation-models-pytorch) with a per-tile presence head
mean-pooled from the encoder bottleneck. Two heads share the backbone per
DESIGN.md §4.3.

Heatmap head: full-resolution 1-channel logit. Trained against a Gaussian
target centered on the labeled pixel; sigmoid is applied at inference time only.

Presence head: scalar logit per tile, derived from a global average pool of the
deepest encoder features (512 channels at 32×32 for ResNet-34 + 1024 px input)
followed by a small MLP. Trained on every tile (positives + negatives).
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

DEFAULT_ENCODER = "resnet34"
DEFAULT_IN_CHANNELS = 4  # chromaticity (3) + wavelength (1)
DEFAULT_PRESENCE_HIDDEN = 128


class LaserDetector(nn.Module):
    """U-Net heatmap + per-tile presence head.

    Args:
        encoder_name: any encoder supported by segmentation-models-pytorch.
        in_channels: input channels (4 for chrom + wavelength).
        encoder_weights: pretrained-weights tag forwarded to smp (e.g. "imagenet"),
            or None for random init.
        presence_hidden: width of the presence MLP hidden layer.
    """

    def __init__(
        self,
        *,
        encoder_name: str = DEFAULT_ENCODER,
        in_channels: int = DEFAULT_IN_CHANNELS,
        encoder_weights: str | None = "imagenet",
        presence_hidden: int = DEFAULT_PRESENCE_HIDDEN,
        decoder_interpolation: str = "nearest",
    ):
        super().__init__()
        # `decoder_interpolation` controls the smp UNet decoder's upsample mode.
        # The default "nearest" matches smp's own default; switching to "bilinear"
        # removes the axis-asymmetric argmax-tie bias that pulls predictions toward
        # smaller (x, y). See notes/bias_attribution.md for the synthetic ablation.
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=1,
            encoder_weights=encoder_weights,
            decoder_interpolation=decoder_interpolation,
        )
        bottleneck_dim = self.unet.encoder.out_channels[-1]
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, presence_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(presence_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.unet.encoder(x)
        bottleneck = features[-1]
        presence_logits = self.presence_head(bottleneck).squeeze(-1)  # [B]
        decoder_output = self.unet.decoder(features)
        heatmap_logits = self.unet.segmentation_head(decoder_output)  # [B, 1, H, W]
        return {
            "heatmap_logits": heatmap_logits,
            "presence_logits": presence_logits,
        }


def line_consistency_loss(
    heatmap_logits: torch.Tensor,    # [B, 1, H, W]
    crop_offsets: torch.Tensor,      # [B, 2] (crop_x, crop_y) in frame coords
    line_abc: torch.Tensor,          # [B, 3] (a, b, c), with (a, b) unit-normalized
    line_confidence: torch.Tensor,   # [B]
    valid_mask: torch.Tensor,        # [B] bool — only these contribute to the loss
    *,
    softmax_temperature: float = 1.0,
) -> torch.Tensor:
    """Per DESIGN.md §5.1: penalize the perpendicular distance from the
    predicted heatmap centroid (soft-argmax) to the dive's RANSAC line.

    Soft-argmax across the (H, W) heatmap gives a differentiable (x, y).
    Adding `crop_offsets` lifts that into frame coordinates so it can be
    compared against the line, which is in frame coordinates too.

    `valid_mask` should be True only for frames where (a) the dive's line is
    confident, and (b) the per-tile presence target is 1 (the label is in
    this crop). Computing soft-argmax on a near-zero heatmap gives a
    near-uniform centroid that has no meaningful relationship to the line.

    `line_abc` are normalized so `a^2 + b^2 = 1`; perpendicular distance
    reduces to `|a*x + b*y + c|`. Returns mean perpendicular distance over
    valid frames, scaled by line_confidence. Returns 0 if no frames are valid.
    """
    if not valid_mask.any():
        return heatmap_logits.new_zeros(())

    B, _, H, W = heatmap_logits.shape
    # Soft-argmax: softmax over the (H*W) flattened spatial map; expectation
    # of x and y under that distribution gives a differentiable centroid.
    flat = (heatmap_logits.view(B, -1) / softmax_temperature)
    soft = torch.softmax(flat, dim=1).view(B, H, W)
    xs = torch.arange(W, device=heatmap_logits.device, dtype=soft.dtype)
    ys = torch.arange(H, device=heatmap_logits.device, dtype=soft.dtype)
    pred_x_local = (soft.sum(dim=1) * xs).sum(dim=1)  # [B]
    pred_y_local = (soft.sum(dim=2) * ys).sum(dim=1)  # [B]

    pred_x = pred_x_local + crop_offsets[:, 0]
    pred_y = pred_y_local + crop_offsets[:, 1]

    a, b, c = line_abc[:, 0], line_abc[:, 1], line_abc[:, 2]
    perp_dist = (a * pred_x + b * pred_y + c).abs()  # [B] frame px

    weight = (line_confidence * valid_mask.float()).clamp(min=0.0)
    n_valid_weight = weight.sum().clamp(min=1.0)
    return (perp_dist * weight).sum() / n_valid_weight


def bce_heatmap_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float = 1000.0,
) -> torch.Tensor:
    """Class-imbalance-corrected BCE for the keypoint heatmap.

    `pred_logits`: [B, 1, H, W] raw logits.
    `target`:      [B, 1, H, W] Gaussian heatmap in [0, 1] (peak ≈ 1.0 at the keypoint).

    Plain BCE collapses on this task: with ~30 non-zero target pixels (within
    3σ of the peak) out of 1M total per tile, the gradient pull from the
    "predict 0 everywhere" minimum dominates. `pos_weight` directly inverts
    that imbalance — each positive pixel's contribution to the loss gets
    multiplied by `pos_weight`, so missing the peak hurts proportionally.

    Default `pos_weight=1000` is roughly 30x the actual pixel ratio
    (~30 active / 1M total). Slightly aggressive on purpose: under-correcting
    fails silently (collapse to 0 redux), over-correcting shows up
    immediately as the heatmap saturating near 1 everywhere (FPR=1.0).

    Returns mean per-pixel loss, matching `binary_cross_entropy_with_logits`'s
    default reduction.
    """
    pw = torch.tensor(pos_weight, device=pred_logits.device, dtype=pred_logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logits, target, pos_weight=pw, reduction="mean",
    )


def focal_heatmap_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CenterNet-style penalty-reduced focal loss for keypoint heatmaps.

    `pred_logits`: [B, 1, H, W] raw logits.
    `target`:      [B, 1, H, W] Gaussian heatmap, peak ≈ 1.0 at the keypoint.

    Reduces the penalty for negative pixels near the peak (where Gaussian
    values are 0 < t < 1) by a `(1 - t)^beta` weight. Frames with all-zero
    targets (negative tiles) reduce to standard focal cross-entropy.

    Returns the mean loss per batch sample (sum over pixels / number of
    "positive" peaks, with a min of 1 to keep it finite for negative tiles).
    """
    pred = torch.sigmoid(pred_logits).clamp(eps, 1.0 - eps)

    pos_mask = target.eq(1.0).float()
    neg_mask = target.lt(1.0).float()

    pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -((1.0 - target) ** beta) * (pred ** alpha) * torch.log(1.0 - pred) * neg_mask

    n_pos = pos_mask.sum()
    if n_pos > 0:
        loss = (pos_loss.sum() + neg_loss.sum()) / n_pos
    else:
        # Negative tile (no peaks): just the negative-pixel term, normalized
        # so its magnitude is comparable to the positive case.
        loss = neg_loss.sum() / max(neg_mask.sum().item(), 1.0)
    return loss
