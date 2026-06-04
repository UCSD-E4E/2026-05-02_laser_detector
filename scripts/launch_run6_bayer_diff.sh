#!/usr/bin/env bash
# Run6 launcher: rebuild bayer_excess_diff cache + train run6_bayer_diff.
#
# Requires:
#   - Kerberos ticket valid (run `kinit` first; check `klist`).
#   - NAS reachable: `ls /home/$USER/mnt/fishsense_data/REEF/data/` should work.
#   - Run5 orchestrator has finished (check chain.log in run5 dir).
#
# Two steps run sequentially, both detached via setsid so they survive
# disconnects:
#   1. Prewarm bayer_excess_diff cache (~2h on 8 workers).
#   2. Train run6_bayer_diff with --bayer-diff-channel + bilinear decoder +
#      --early-stop-patience 10. Auto-fires after the prewarm finishes.
#
# Usage:
#   bash scripts/launch_run6_bayer_diff.sh
#
# Logs:
#   - data/image_cache_bayer_excess_diff/prewarm.log
#   - data/phase2/checkpoints_sensor_bayer_50e_run6_bayer_diff/train.log
#   - data/phase2/checkpoints_sensor_bayer_50e_run6_bayer_diff/chain.log
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR=$REPO/data/image_cache_bayer_excess_diff
RUN6_DIR=$REPO/data/phase2/checkpoints_sensor_bayer_50e_run6_bayer_diff
mkdir -p "$CACHE_DIR" "$RUN6_DIR"

# Sanity check NAS access
if ! ls "/home/$USER/mnt/fishsense_data/REEF/data/" > /dev/null 2>&1; then
  echo "ERROR: NAS not reachable. Run 'kinit' first and verify 'ls' works." >&2
  exit 1
fi
echo "NAS access OK"

# Launch chained: prewarm → train, both detached
cd "$REPO"
nix develop --command bash -c "
  setsid --fork bash -c '
    echo \"=== run6 chain start \$(date) pid=\$\$ — prewarming bayer_excess_diff cache ===\" >> $RUN6_DIR/chain.log
    uv run python scripts/prewarm_bayer_excess_cache.py \
      --splits train val test \
      --with-diff-channel \
      --cache-dir $CACHE_DIR \
      --workers 8 > $CACHE_DIR/prewarm.log 2>&1
    PREWARM_RC=\$?
    echo \"=== prewarm rc=\$PREWARM_RC \$(date) ===\" >> $RUN6_DIR/chain.log
    if [ \$PREWARM_RC -ne 0 ]; then
      if ! grep -q \"Done in\" $CACHE_DIR/prewarm.log 2>/dev/null; then
        echo \"=== prewarm did NOT complete cleanly; aborting run6 launch ===\" >> $RUN6_DIR/chain.log
        exit 1
      fi
    fi
    echo \"=== launching run6_bayer_diff training \$(date) ===\" >> $RUN6_DIR/chain.log
    echo \"=== run6_bayer_diff start \$(date) pid=\$\$ ===\" >> $RUN6_DIR/train.log
    uv run torchrun --standalone --nproc_per_node=4 scripts/run_train.py \
      --image-pipeline linear_npy \
      --bayer-excess \
      --bayer-diff-channel \
      --bayer-excess-cache-dir $CACHE_DIR \
      --decoder-interpolation bilinear \
      --num-workers 4 \
      --early-stop-patience 10 \
      --checkpoint-dir $RUN6_DIR \
      --resume auto \
      --no-mlflow \
      >> $RUN6_DIR/train.log 2>&1
    echo \"=== run6_bayer_diff exited rc=\$? \$(date) ===\" >> $RUN6_DIR/chain.log
  ' &
"
sleep 2
echo
echo "=== launched ==="
echo "Prewarm logs:  $CACHE_DIR/prewarm.log"
echo "Train logs:    $RUN6_DIR/train.log"
echo "Chain status:  $RUN6_DIR/chain.log"
echo
echo "Expected timing:"
echo "  prewarm: ~2h (33k ORF decodes, 8 workers)"
echo "  train:   ~24h worst case, ~14h with early-stop"
