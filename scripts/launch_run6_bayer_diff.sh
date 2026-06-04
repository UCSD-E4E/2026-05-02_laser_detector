#!/usr/bin/env bash
# Run6 launcher: rebuild bayer_excess_diff cache + train run6_bayer_diff.
#
# Steps run sequentially, detached via setsid, wrapped in krenew -K 60 so
# the Kerberos ticket gets re-acquired every 60 min throughout the run.
#   1. Prewarm bayer_excess_diff cache (~2h on 8 workers, needs NAS).
#   2. Train run6_bayer_diff with --bayer-diff-channel + bilinear decoder +
#      --early-stop-patience 10 (~14-24h).
#
# Requires:
#   - Kerberos ticket valid AND renewable (kinit beforehand, klist to
#     confirm "renew until" is in the future).
#   - NAS reachable: `ls /home/$USER/mnt/fishsense_data/REEF/data/` works.
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

# Launch chained: krenew -K 60 -- (prewarm; train), detached
cd "$REPO"
nix develop --command bash -c "
  setsid --fork nix shell nixpkgs#kstart --command krenew -K 60 -- bash $REPO/scripts/run6_chain.sh &
"
sleep 3
echo
echo "=== launched ==="
echo "Prewarm logs:  $CACHE_DIR/prewarm.log"
echo "Train logs:    $RUN6_DIR/train.log"
echo "Chain status:  $RUN6_DIR/chain.log"
echo
echo "Expected timing:"
echo "  prewarm: ~2h (33k ORF decodes, 8 workers)"
echo "  train:   ~24h worst case, ~14h with early-stop"
echo
echo "krenew is renewing every 60 min while the chain runs."
