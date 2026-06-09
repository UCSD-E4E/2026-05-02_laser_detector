#!/usr/bin/env bash
# Run8: HRNet-w18 + smaller σ (1.5 vs default 3.0).
# Tests sub-pixel-aware training - does sharper supervision tighten the
# borderline-precision mode that the parabolic peak fit can't close?
set -euo pipefail
REPO=/scratch/krg/c.crutchfield.642/Repos/school/e4e/fishsense/2026-05-02_laser_detector
CACHE_DIR=$REPO/data/image_cache_bayer_excess
RUN8_DIR=$REPO/data/phase2/checkpoints_sensor_bayer_50e_run8_hrnet_w18_sigma15
mkdir -p "$RUN8_DIR"
cd "$REPO"

echo "Cache coverage check: skipped (NAS not required for cached splits)."

echo "=== run8 hrnet_w18 sigma=1.5 start $(date) ===" >> "$RUN8_DIR/chain.log"
echo "=== run8 hrnet_w18 sigma=1.5 start $(date) ===" >> "$RUN8_DIR/train.log"
nix develop --command uv run torchrun --standalone --nproc_per_node=4 scripts/run_train.py \
  --image-pipeline linear_npy \
  --bayer-excess \
  --bayer-excess-cache-dir "$CACHE_DIR" \
  --encoder-name tu-hrnet_w18 \
  --heatmap-sigma-px 1.5 \
  --batch-size 8 \
  --num-workers 4 \
  --early-stop-patience 10 \
  --checkpoint-dir "$RUN8_DIR" \
  --resume auto \
  --no-mlflow \
  >> "$RUN8_DIR/train.log" 2>&1
TRAIN_RC=$?
echo "=== run8 exited rc=$TRAIN_RC $(date) ===" >> "$RUN8_DIR/chain.log"
exit $TRAIN_RC
