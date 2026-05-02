"""Phase 0 configuration loaded via Dynaconf.

Layered config (later tiers override earlier ones):
1. `settings.toml`         (committed) — defaults for non-secret settings.
2. `settings.local.toml`   (gitignored) — user-local non-secret overrides
                             (e.g. filesystem paths). Optional. Template at
                             `settings.local.toml.example`.
3. `.secrets.toml`         (gitignored) — API credentials. Template at
                             `.secrets.toml.example`.
4. Environment variables prefixed with `LASER_` — override all of the above.
                             Use `__` for nested keys, e.g.
                             `LASER_RUN__MAX_DIVES=5`.

The runtime view is a frozen `Phase0Config` dataclass so the rest of the
pipeline gets type-checked attribute access rather than dict lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dynaconf import Dynaconf, Validator

# Module-level Dynaconf object. Files are resolved relative to the project root
# (current working directory when scripts are invoked via `uv run`).
settings = Dynaconf(
    envvar_prefix="LASER",
    settings_files=["settings.toml", "settings.local.toml", ".secrets.toml"],
    environments=False,
    load_dotenv=False,
    merge_enabled=True,
    validators=[
        Validator(
            "api.base_url",
            must_exist=True,
            is_type_of=str,
            condition=lambda v: bool(v),
            messages={"condition": "api.base_url must be set in settings.toml"},
        ),
        Validator(
            "api.username",
            must_exist=True,
            is_type_of=str,
            condition=lambda v: bool(v),
            messages={"condition": "api.username must be set in .secrets.toml"},
        ),
        Validator(
            "api.password",
            must_exist=True,
            is_type_of=str,
            condition=lambda v: bool(v),
            messages={"condition": "api.password must be set in .secrets.toml"},
        ),
        Validator(
            "mlflow.tracking_uri",
            must_exist=True,
            is_type_of=str,
            condition=lambda v: bool(v),
            messages={"condition": "mlflow.tracking_uri must be set in settings.toml"},
        ),
        Validator(
            "mlflow.username",
            must_exist=True,
            is_type_of=str,
            condition=lambda v: bool(v),
            messages={"condition": "mlflow.username must be set in .secrets.toml"},
        ),
        Validator(
            "mlflow.token",
            must_exist=True,
            is_type_of=str,
            condition=lambda v: bool(v),
            messages={"condition": "mlflow.token must be set in .secrets.toml"},
        ),
        Validator("api.max_concurrent_requests", is_type_of=int, gte=1),
        Validator("api.timeout_seconds", is_type_of=int, gte=1),
        Validator("data.dir", must_exist=True, is_type_of=str),
        # images.root may be empty (no image loading); just type-check it.
        Validator("images.root", is_type_of=str),
        Validator("cache.dir", must_exist=True, is_type_of=str),
        Validator(
            "cache.jpeg_quality", is_type_of=int, gte=1, lte=100
        ),
        Validator("run.canonical_only", is_type_of=bool),
        Validator("run.max_dives", is_type_of=int, gte=0),
        Validator("run.image_workers", is_type_of=int, gte=1),
        Validator("audit.sample_dives", is_type_of=int, gte=0),
        Validator("audit.samples_per_dive", is_type_of=int, gte=0),
        Validator("splits.seed", is_type_of=int),
        Validator(
            "splits.train_frac", is_type_of=float, gt=0.0, lt=1.0
        ),
        Validator("splits.val_frac", is_type_of=float, gt=0.0, lt=1.0),
        Validator("rng_seed", is_type_of=int),
    ],
)


@dataclass(frozen=True)
class Phase0Config:
    """Typed view over the Dynaconf-loaded settings."""

    # Fishsense API
    api_base_url: str
    api_username: str
    api_password: str
    api_max_concurrent_requests: int
    api_timeout_seconds: int

    # Where preprocessing artifacts go
    data_dir: Path

    # Image source root (None = no image loading)
    image_root: Path | None

    # Image-decode cache
    cache_dir: Path
    cache_jpeg_quality: int

    # Run controls
    max_dives: int | None
    canonical_only: bool
    image_workers: int

    # Laser-size audit
    audit_sample_dives: int
    audit_samples_per_dive: int

    # Splits
    split_seed: int
    split_train_frac: float
    split_val_frac: float

    # Reproducibility
    rng_seed: int

    # MLflow tracking server (basic auth via mlflow-oidc-auth plugin)
    mlflow_tracking_uri: str
    mlflow_username: str
    mlflow_token: str


def load_config() -> Phase0Config:
    """Validate and materialize a Phase0Config from Dynaconf settings.

    Raises a Dynaconf `ValidationError` if a required setting is missing
    or has the wrong type. The error message names the missing key, so
    fixing it is mechanical.
    """
    settings.validators.validate()

    raw_max_dives = int(settings["run.max_dives"])
    return Phase0Config(
        api_base_url=str(settings["api.base_url"]),
        api_username=str(settings["api.username"]),
        api_password=str(settings["api.password"]),
        api_max_concurrent_requests=int(settings["api.max_concurrent_requests"]),
        api_timeout_seconds=int(settings["api.timeout_seconds"]),
        data_dir=Path(str(settings["data.dir"])),
        image_root=(
            Path(str(settings["images.root"]))
            if settings.get("images.root")
            else None
        ),
        cache_dir=Path(str(settings["cache.dir"])),
        cache_jpeg_quality=int(settings["cache.jpeg_quality"]),
        max_dives=raw_max_dives if raw_max_dives > 0 else None,
        canonical_only=bool(settings["run.canonical_only"]),
        image_workers=int(settings["run.image_workers"]),
        audit_sample_dives=int(settings["audit.sample_dives"]),
        audit_samples_per_dive=int(settings["audit.samples_per_dive"]),
        split_seed=int(settings["splits.seed"]),
        split_train_frac=float(settings["splits.train_frac"]),
        split_val_frac=float(settings["splits.val_frac"]),
        rng_seed=int(settings["rng_seed"]),
        mlflow_tracking_uri=str(settings["mlflow.tracking_uri"]),
        mlflow_username=str(settings["mlflow.username"]),
        mlflow_token=str(settings["mlflow.token"]),
    )
