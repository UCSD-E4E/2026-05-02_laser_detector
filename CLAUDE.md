# Claude instructions — laser detector

A post-processing detector for the laser dot in fishsense dive imagery. **`DESIGN.md` is the source of truth** for what we're building. Read it before making non-trivial changes.

## Tooling

- **Environment**: `uv` (Astral). `pyproject.toml` is authoritative, `uv.lock` is checked in.
  - Add deps: `uv add <pkg>` (or `uv add --dev <pkg>` for dev deps).
  - Run anything: `uv run <cmd>` — do not assume an activated venv.
  - Do **not** create `requirements.txt`, `setup.py`, or `environment.yml`.
- **Python**: 3.13 (see `.python-version`).
- **ML framework**: **PyTorch**. Not TensorFlow, not JAX. Default ecosystem:
  - `segmentation-models-pytorch` for U-Net + ResNet-34 encoder.
  - `timm` for backbones not in `smp`.
  - `albumentations` for augmentation (handles keypoint coords correctly).
  - `torchmetrics` for metrics.
  - `mlflow` for tracking and registry.
- **Data access**: `fishsense-sdk` (`UCSD-E4E/fishsense-lite` on GitHub). Labels and frames flow through it.
- **Image decode**: `fishsense-core` (`UCSD-E4E/fishsense-core` wheels). ORF files (Olympus TG-6) decode via `fishsense_core.image.raw_image.RawImage` to match the project's standard pipeline (rawpy → auto-gamma → CLAHE). **Don't use rawpy directly** — input distribution must match other models.
- **Image cache**: `CachingImageLoader` writes JPEGs (keyed by checksum) to `data/image_cache/` since ORF decode is slow. Wraps any inner loader; gitignored.

## Things easy to get wrong

- **Splits are dive-level, never frame-level.** Frame-level leaks the per-dive line and wavelength priors into validation. Stratify by wavelength.
- **Input is tiled, not downsampled.** Native frames are 4K and the laser is 3–8 px. Any meaningful downscale loses the target. Use 1024×1024 tiles at native resolution with 256 px overlap (see DESIGN §4.1).
- **Evaluation tolerance is the on-screen laser blob**, not pixel distance to the label centroid. The label is a point; correctness is "did we land inside the laser dot." Fixed-tolerance fallback is `N=3` strict / `N=4` lenient.
- **Geometric augmentation is off by default.** Rotation and flip break the per-dive colinearity prior. Photometric augs (hue, brightness, blur, noise) are fine and important.
- **Negative frames are real training data**, not just an inference concern. Train the presence head with hard-negative mining, not random sampling.
- **Wavelength is recovered offline by clustering per-dive colors at labels.** It's not in the SDK. Don't ask the user; derive it.
- **Future rig changes get a separate model.** Tag `rig_id` from Phase 0; default to `rig=v1`.

## Documents in this repo

- `GOAL.md` — original problem framing and observations from the user.
- `DESIGN.md` — current architecture, training plan, MLflow integration, phased rollout.
- `README.md` — human-facing summary and quick start.
- `CLAUDE.md` — this file.

## When in doubt

- Don't introduce ViT, RL, DETR, or other heavy machinery without first checking DESIGN.md — that ground was already covered and rejected for this problem.
- Don't add comments explaining what code does. Only WHY, when non-obvious.
- Don't over-refactor: bug fixes don't need surrounding cleanup.
- Ask before destructive operations (deleting data, force-pushing, dropping artifacts from MLflow).
