#!/usr/bin/env bash
# Run7: HRNet-w32 encoder retrain, same data + recipe as run3.
# Tests whether HRNet's inductive bias helps the borderline-precision mode
# (70% of remaining test failures are 3-10 px misses).
#
# Caveat: smp.Unet wraps HRNet's encoder into standard 5-stage downsampling,
# so this isn't "true" HRNet (high-res branches). Modest expected benefit.
# A custom timm-HRNet + heatmap-head integration is the follow-up if this
# doesn't move the needle.
set -euo pipefail
REPO=/scratch/krg/c.crutchfield.642/Repos/school/e4e/fishsense/2026-05-02_laser_detector
CACHE_DIR=$REPO/data/image_cache_bayer_excess
RUN7_DIR=$REPO/data/phase2/checkpoints_sensor_bayer_50e_run7_hrnet_w32
mkdir -p "$RUN7_DIR"
cd "$REPO"

# NAS not required: train 99.9% / val 94.2% / test 100% cached for
# bayer_excess. The ~230 missing val frames are the known dive 249 stale
# paths; loader returns None and the trainer drops them (same as every
# prior run). Kerberos can be expired and this still trains cleanly.
echo "Cache coverage: train ~100% cached; proceeding without NAS check."

echo "=== run7 hrnet_w32 start $(date) ===" >> "$RUN7_DIR/chain.log"
echo "=== run7 hrnet_w32 start $(date) ===" >> "$RUN7_DIR/train.log"
nix develop --command uv run torchrun --standalone --nproc_per_node=4 scripts/run_train.py \
  --image-pipeline linear_npy \
  --bayer-excess \
  --bayer-excess-cache-dir "$CACHE_DIR" \
  --encoder-name tu-hrnet_w32 \
  --num-workers 4 \
  --early-stop-patience 10 \
  --checkpoint-dir "$RUN7_DIR" \
  --resume auto \
  --no-mlflow \
  >> "$RUN7_DIR/train.log" 2>&1
TRAIN_RC=$?
echo "=== run7 exited rc=$TRAIN_RC $(date) ===" >> "$RUN7_DIR/chain.log"
exit $TRAIN_RC
