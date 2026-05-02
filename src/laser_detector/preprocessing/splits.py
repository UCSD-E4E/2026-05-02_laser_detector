"""Dive-level train/val/test splits, stratified by wavelength.

Splits are at the dive level — frame-level splits would leak the per-dive line
and wavelength priors into validation. Stratification by wavelength ensures
both green and blue dives are present in each split.

Output:
    one row per dive: dive_id, split ("train" | "val" | "test")
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


SPLIT_TABLE_SCHEMA = {
    "dive_id": pl.Int64,
    "split": pl.Utf8,
}


def make_dive_splits(
    wavelengths: pl.DataFrame,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> pl.DataFrame:
    """Assign each dive to train/val/test, stratified by wavelength."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to <1")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # Group by wavelength so dives with unknown wavelength still get split
    # (their own group, deterministic).
    groups = wavelengths.group_by("wavelength", maintain_order=True)
    for (wavelength,), group in groups:
        dive_ids = group["dive_id"].to_list()
        rng.shuffle(dive_ids)
        n = len(dive_ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Adjust if rounding overshot
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_ids = dive_ids[:n_train]
        val_ids = dive_ids[n_train : n_train + n_val]
        test_ids = dive_ids[n_train + n_val :]

        for dive_id in train_ids:
            rows.append({"dive_id": int(dive_id), "split": "train"})
        for dive_id in val_ids:
            rows.append({"dive_id": int(dive_id), "split": "val"})
        for dive_id in test_ids:
            rows.append({"dive_id": int(dive_id), "split": "test"})

        logger.info(
            "Wavelength=%s: %d dives → %d train, %d val, %d test",
            wavelength,
            n,
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )

    return pl.DataFrame(rows, schema=SPLIT_TABLE_SCHEMA)
