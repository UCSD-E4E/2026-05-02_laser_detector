"""Phase 0 orchestration: ingest → priors → audit → splits.

Each step writes a parquet file to `config.data_dir`. Re-running picks up
existing intermediate files unless `force=True`. This keeps iteration cheap —
the slow step (ingest) only runs when needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from laser_detector.preprocessing.audit import laser_size_audit
from laser_detector.preprocessing.config import Phase0Config
from laser_detector.preprocessing.image_loader import ImageLoader, NullImageLoader
from laser_detector.preprocessing.ingest import build_frame_table
from laser_detector.preprocessing.line_fit import (
    fit_lines_per_dive,
    flag_label_outliers,
)
from laser_detector.preprocessing.splits import make_dive_splits
from laser_detector.preprocessing.wavelength import infer_wavelengths

logger = logging.getLogger(__name__)


# Output filenames inside config.data_dir
FRAMES_FILE = "frames.parquet"
LINES_FILE = "dive_lines.parquet"
WAVELENGTHS_FILE = "dive_wavelengths.parquet"
AUDIT_FILE = "laser_size_audit.parquet"
SPLITS_FILE = "dive_splits.parquet"
FRAMES_FLAGGED_FILE = "frames_with_outliers.parquet"


def _read_or_compute(
    path: Path,
    compute_fn,
    *,
    force: bool,
    label: str,
) -> pl.DataFrame:
    if path.exists() and not force:
        logger.info("Reading cached %s from %s", label, path)
        return pl.read_parquet(path)
    df = compute_fn()
    df.write_parquet(path)
    logger.info("Wrote %s (%d rows) to %s", label, df.height, path)
    return df


def run_phase0(
    config: Phase0Config,
    image_loader: ImageLoader | None = None,
    *,
    force: bool = False,
) -> dict[str, pl.DataFrame]:
    """Run the full Phase 0 pipeline. Returns the artifacts as a dict."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    image_loader = image_loader or NullImageLoader()

    # 1. Frame table (from SDK)
    frames = _read_or_compute(
        config.data_dir / FRAMES_FILE,
        lambda: build_frame_table(config),
        force=force,
        label="frame table",
    )

    # 2. Per-dive line fits + outlier flagging
    lines = _read_or_compute(
        config.data_dir / LINES_FILE,
        lambda: fit_lines_per_dive(frames, seed=config.split_seed),
        force=force,
        label="dive lines",
    )
    frames_flagged = _read_or_compute(
        config.data_dir / FRAMES_FLAGGED_FILE,
        lambda: flag_label_outliers(frames, lines),
        force=force,
        label="frames with outlier flags",
    )

    # 3. Per-dive wavelength inference (needs image bytes; falls back to label_string)
    wavelengths = _read_or_compute(
        config.data_dir / WAVELENGTHS_FILE,
        lambda: infer_wavelengths(frames, image_loader),
        force=force,
        label="dive wavelengths",
    )

    # 4. Laser-size audit (needs image bytes)
    audit = _read_or_compute(
        config.data_dir / AUDIT_FILE,
        lambda: laser_size_audit(
            frames,
            wavelengths,
            image_loader,
            sample_dives=config.audit_sample_dives,
            samples_per_dive=config.audit_samples_per_dive,
            rng_seed=config.rng_seed,
        ),
        force=force,
        label="laser-size audit",
    )

    # 5. Dive-level splits stratified by wavelength
    splits = _read_or_compute(
        config.data_dir / SPLITS_FILE,
        lambda: make_dive_splits(
            wavelengths,
            train_frac=config.split_train_frac,
            val_frac=config.split_val_frac,
            seed=config.split_seed,
        ),
        force=force,
        label="dive splits",
    )

    return {
        "frames": frames,
        "frames_flagged": frames_flagged,
        "lines": lines,
        "wavelengths": wavelengths,
        "audit": audit,
        "splits": splits,
    }
