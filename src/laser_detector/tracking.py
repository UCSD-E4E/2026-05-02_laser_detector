"""MLflow client setup for runs originating from this project.

Reads the tracking URI + basic-auth credentials from `Phase0Config` (loaded
via Dynaconf from `settings.toml` + `.secrets.toml`), exports the env vars
the mlflow Python client picks up automatically, points the client at the
configured tracking server, and selects (creating if needed) the canonical
project experiment.

The server uses the `mlflow-oidc-auth` plugin in basic-auth mode: the
"username" is the user's OIDC email, the "password" is the API token
generated from the user's profile page on the MLflow UI.
"""

from __future__ import annotations

import logging
import os

import mlflow

from laser_detector.preprocessing.config import Phase0Config

logger = logging.getLogger(__name__)

# Canonical experiment name for this project. All runs (Phase 1 baseline,
# Phase 2+ supervised) land here; per-phase grouping is done via tags.
EXPERIMENT_NAME = "2026-05-02_laser_detector"


def setup_mlflow(config: Phase0Config) -> str:
    """Configure mlflow for this process and return the experiment_id.

    Idempotent — safe to call multiple times.
    """
    os.environ["MLFLOW_TRACKING_USERNAME"] = config.mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config.mlflow_token
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(
        "MLflow ready: tracking_uri=%s experiment=%r id=%s",
        config.mlflow_tracking_uri,
        EXPERIMENT_NAME,
        experiment.experiment_id,
    )
    return experiment.experiment_id
