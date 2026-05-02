"""Pull dives, images, and laser labels from the fishsense API into a frame table.

The frame-level table is the canonical input to every other Phase 0 step. One row
per labeled image. Columns: dive_id, image_id, rig_id (= camera_id), image_path,
image_checksum, label_x, label_y, is_positive, label_studio_*, label_string,
superseded, completed.

Negative frames (no laser visible) are included — `LaserLabel` rows with x/y
non-null are positives; the API also supports recording explicit negatives via
laser labels with the `label` field set accordingly. We treat a row as positive
iff both `x` and `y` are non-null and `superseded` is not True.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import polars as pl
from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from tqdm.asyncio import tqdm_asyncio

from laser_detector.preprocessing.config import Phase0Config

logger = logging.getLogger(__name__)


FRAME_TABLE_SCHEMA = {
    "dive_id": pl.Int64,
    "image_id": pl.Int64,
    "rig_id": pl.Int64,  # camera_id
    "image_path": pl.Utf8,
    "image_checksum": pl.Utf8,
    "label_x": pl.Float64,
    "label_y": pl.Float64,
    "is_positive": pl.Boolean,
    "label_string": pl.Utf8,  # `label` field on LaserLabel; may carry color or "no laser" text
    "label_studio_task_id": pl.Int64,
    "label_studio_project_id": pl.Int64,
    "superseded": pl.Boolean,
    "completed": pl.Boolean,
}


@dataclass
class _DiveBundle:
    dive: Dive
    images: list[Image]
    laser_labels: list[LaserLabel]


async def _fetch_dive_bundle(client: Client, dive: Dive) -> _DiveBundle | None:
    """Fetch the images and laser labels for one dive concurrently.

    Catches per-dive errors so a single bad request doesn't tear down the
    whole pipeline. A warning is logged and `None` is returned; the caller
    filters those out.
    """
    if dive.id is None:
        return None
    try:
        images_task = client.images.get(dive_id=dive.id)
        labels_task = client.labels.get_laser_labels(dive_id=dive.id)
        images, labels = await asyncio.gather(images_task, labels_task)
    except Exception as exc:
        logger.warning("Failed to fetch dive %s: %s: %s", dive.id, type(exc).__name__, exc)
        return None
    return _DiveBundle(
        dive=dive,
        images=list(images) if images else [],
        laser_labels=list(labels) if labels else [],
    )


async def _list_dives(client: Client, config: Phase0Config) -> list[Dive]:
    if config.canonical_only:
        dives = await client.dives.get_canonical()
    else:
        dives = await client.dives.get()
    if dives is None:
        return []
    if isinstance(dives, Dive):
        dives = [dives]
    if config.max_dives is not None:
        dives = dives[: config.max_dives]
    return dives


def _bundles_to_rows(bundles: list[_DiveBundle]) -> list[dict]:
    """Flatten dive bundles into per-image rows, joining labels to images by image_id.

    An image without a laser label is skipped — without a label we can't say
    whether it's a true positive or negative. The 60k-label corpus is the dataset.
    """
    rows: list[dict] = []
    for bundle in bundles:
        rig_id = bundle.dive.camera_id
        images_by_id = {img.id: img for img in bundle.images if img.id is not None}
        for label in bundle.laser_labels:
            if label.image_id is None:
                continue
            image = images_by_id.get(label.image_id)
            if image is None:
                continue
            x, y = label.x, label.y
            is_positive = (
                x is not None
                and y is not None
                and not bool(label.superseded)
            )
            rows.append(
                {
                    "dive_id": bundle.dive.id,
                    "image_id": image.id,
                    "rig_id": rig_id,
                    "image_path": image.path,
                    "image_checksum": image.checksum,
                    "label_x": float(x) if x is not None else None,
                    "label_y": float(y) if y is not None else None,
                    "is_positive": is_positive,
                    "label_string": label.label,
                    "label_studio_task_id": label.label_studio_task_id,
                    "label_studio_project_id": label.label_studio_project_id,
                    "superseded": bool(label.superseded) if label.superseded is not None else None,
                    "completed": bool(label.completed) if label.completed is not None else None,
                }
            )
    return rows


async def build_frame_table_async(config: Phase0Config) -> pl.DataFrame:
    """Async build of the frame-level table by pulling everything in parallel."""
    async with Client(
        base_url=config.api_base_url,
        username=config.api_username,
        password=config.api_password,
        timeout=config.api_timeout_seconds,
        max_concurrent_requests=config.api_max_concurrent_requests,
    ) as client:
        dives = await _list_dives(client, config)
        logger.info("Fetched %d dives", len(dives))

        # `_fetch_dive_bundle` returns None on failure (logged inside), so
        # one bad dive doesn't poison the whole gather. The SDK's internal
        # semaphore caps in-flight requests at `max_concurrent_requests`.
        bundle_tasks = [_fetch_dive_bundle(client, dive) for dive in dives]
        bundles_with_none = await tqdm_asyncio.gather(*bundle_tasks, desc="dives")
        bundles = [b for b in bundles_with_none if b is not None]
        n_failed = len(bundles_with_none) - len(bundles)
        if n_failed:
            logger.warning(
                "Failed to fetch %d / %d dives — see warnings above",
                n_failed,
                len(bundles_with_none),
            )

    rows = _bundles_to_rows(bundles)
    df = pl.DataFrame(rows, schema=FRAME_TABLE_SCHEMA)
    logger.info(
        "Built frame table: %d rows, %d positives, %d dives, %d rigs",
        df.height,
        df.filter(pl.col("is_positive")).height,
        df["dive_id"].n_unique(),
        df["rig_id"].n_unique(),
    )
    return df


def build_frame_table(config: Phase0Config) -> pl.DataFrame:
    """Sync wrapper around :func:`build_frame_table_async`."""
    return asyncio.run(build_frame_table_async(config))
