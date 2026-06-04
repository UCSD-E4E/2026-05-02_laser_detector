#!/usr/bin/env bash
# Inner chain script for run6_bayer_diff: prewarm cache, then train.
# Designed to be wrapped in `krenew -K 60 -- bash run6_chain.sh` so the
# Kerberos ticket gets renewed every 60 minutes throughout the ~26h run.
set -euo pipefail

REPO=/scratch/krg/c.crutchfield.642/Repos/school/e4e/fishsense/2026-05-02_laser_detector
CACHE_DIR=$REPO/data/image_cache_bayer_excess_diff
RUN6_DIR=$REPO/data/phase2/checkpoints_sensor_bayer_50e_run6_bayer_diff
cd "$REPO"

echo "=== prewarm start $(date) pid=$$ ===" >> "$RUN6_DIR/chain.log"
# The prewarm script returns 1 if any frame fails to decode, but ~230 expected
# failures (dive 249 stale paths + 19 train misses) are normal. Use `|| true`
# to keep the chain alive past `set -e`; check the "Done in" marker instead.
uv run python scripts/prewarm_bayer_excess_cache.py \
  --splits train val test \
  --with-diff-channel \
  --cache-dir "$CACHE_DIR" \
  --workers 8 \
  > "$CACHE_DIR/prewarm.log" 2>&1 || true
PREWARM_RC=$?
echo "=== prewarm rc=$PREWARM_RC $(date) ===" >> "$RUN6_DIR/chain.log"
if ! grep -q "Done in" "$CACHE_DIR/prewarm.log" 2>/dev/null; then
  echo "=== prewarm did NOT complete cleanly (no 'Done in' marker); aborting $(date) ===" >> "$RUN6_DIR/chain.log"
  exit 1
fi

echo "=== launching run6_bayer_diff training $(date) ===" >> "$RUN6_DIR/chain.log"
echo "=== run6_bayer_diff start $(date) ===" >> "$RUN6_DIR/train.log"
uv run torchrun --standalone --nproc_per_node=4 scripts/run_train.py \
  --image-pipeline linear_npy \
  --bayer-excess \
  --bayer-diff-channel \
  --bayer-excess-cache-dir "$CACHE_DIR" \
  --decoder-interpolation bilinear \
  --num-workers 4 \
  --early-stop-patience 10 \
  --checkpoint-dir "$RUN6_DIR" \
  --resume auto \
  --no-mlflow \
  >> "$RUN6_DIR/train.log" 2>&1
TRAIN_RC=$?
echo "=== run6_bayer_diff exited rc=$TRAIN_RC $(date) ===" >> "$RUN6_DIR/chain.log"
exit $TRAIN_RC
