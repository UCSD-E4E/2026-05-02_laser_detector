"""Fetch per-rig camera intrinsics from the fishsense API and persist them
locally for use at inference. Run this ONCE with valid API credentials; the
resulting parquet is checkpoint-agnostic and rig-set-scoped.

Usage:
    nix develop --command uv run python scripts/ingest_camera_intrinsics.py \\
      --frames data/frames.parquet \\
      --out data/rig_intrinsics.parquet

Output schema:
    rig_id (int)
    fx, fy, cx, cy (float) — camera matrix components
    dist (list[float]) — distortion coefficients (typ. 5-element k1,k2,p1,p2,k3)

This addresses issue #9: labels live in rectified coord space, predictions
land in raw image coord space, and downstream 3D reconstruction needs a
consistent frame. The eval/audit scripts can then apply cv2.undistortPoints
at inference to move predictions from raw → rectified.

Empirical impact is tiny (median displacement 0.02 px, p99 1.01 px per
issue #9) so no retrain is needed — just apply this shift at inference.
"""
from __future__ import annotations
import argparse
import asyncio
import getpass
import logging
import os
from pathlib import Path
import sys

import numpy as np
import polars as pl
from fishsense_api_sdk.client import Client

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from laser_detector.preprocessing.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)


async def fetch_intrinsics(rig_ids: list[int], base_url: str, username: str, password: str) -> list[dict]:
    """Fetch per-rig intrinsics; skip missing rigs with a warning."""
    rows: list[dict] = []
    async with Client(base_url, username, password) as client:
        for rig_id in rig_ids:
            try:
                intr = await client.cameras.get_intrinsics(camera_id=rig_id)
            except Exception as e:
                logger.warning('rig_id=%d: fetch failed (%s: %s); skipping', rig_id, type(e).__name__, e)
                continue
            if intr is None:
                logger.warning('rig_id=%d: no intrinsics registered; skipping', rig_id)
                continue
            K = np.asarray(intr.camera_matrix, dtype=float)
            dist = (
                np.asarray(intr.distortion_coefficients, dtype=float).ravel().tolist()
                if intr.distortion_coefficients is not None else []
            )
            rows.append({
                'rig_id': rig_id,
                'fx': float(K[0, 0]),
                'fy': float(K[1, 1]),
                'cx': float(K[0, 2]),
                'cy': float(K[1, 2]),
                'dist': dist,
            })
            logger.info('rig_id=%d: fx=%.1f fy=%.1f cx=%.1f cy=%.1f dist=%s',
                        rig_id, K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist)
    return rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=Path, default=None,
                   help='frames.parquet to pull unique rig_ids from. Defaults to config.data_dir/frames.parquet.')
    p.add_argument('--out', type=Path, default=None,
                   help='Output parquet path. Defaults to config.data_dir/rig_intrinsics.parquet.')
    p.add_argument('--rig-ids', type=int, nargs='+', default=None,
                   help='Override rig_ids to fetch instead of reading from frames.')
    args = p.parse_args()

    cfg = load_config()
    frames_path = args.frames or (cfg.data_dir / 'frames.parquet')
    out_path = args.out or (cfg.data_dir / 'rig_intrinsics.parquet')

    if args.rig_ids is not None:
        rig_ids = sorted(set(args.rig_ids))
    else:
        frames = pl.read_parquet(frames_path)
        rig_ids = sorted(frames['rig_id'].drop_nulls().unique().to_list())
    logger.info('fetching intrinsics for rig_ids=%s', rig_ids)

    base_url = cfg.api_base_url
    username = os.environ.get('FISHSENSE_USERNAME') or input('fishsense username: ')
    password = os.environ.get('FISHSENSE_PASSWORD') or getpass.getpass('fishsense password: ')

    rows = asyncio.run(fetch_intrinsics(rig_ids, base_url, username, password))
    if not rows:
        logger.error('no intrinsics fetched; aborting')
        return 1
    df = pl.DataFrame(rows, schema={
        'rig_id': pl.Int64, 'fx': pl.Float64, 'fy': pl.Float64,
        'cx': pl.Float64, 'cy': pl.Float64, 'dist': pl.List(pl.Float64),
    })
    df.write_parquet(out_path)
    logger.info('wrote %d rigs to %s', df.height, out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
