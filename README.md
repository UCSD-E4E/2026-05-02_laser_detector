# Laser Detector

Post-processing detector for the laser dot in fishsense dive imagery. Given a frame, returns either `(x, y)` of the laser center or `no_laser`.

Part of the [UCSD E4E fishsense](https://github.com/UCSD-E4E/fishsense-lite) project. Real-time inference on a UUV is the eventual deployment target; v1 targets server-side batch processing on still images.

## Status

Early. Architecture is designed; nothing is trained yet. See [DESIGN.md](DESIGN.md) for the full plan and [GOAL.md](GOAL.md) for the original problem framing.

## Approach in one paragraph

A U-Net (ResNet-34 encoder) heatmap detector with a presence head, trained on ~60k labels from `fishsense-sdk`. Input is tiled at native 4K resolution because the laser is only 3–8 px across — downsampling loses the target. Two per-dive priors are recovered offline and used to clean labels and constrain inference: a colinearity line fit (the laser rig is fixed), and a green/blue wavelength tag (each dive is single-color, but the wavelength field isn't recorded). Splits are at the dive level so the priors don't leak into validation.

## Quick start

```bash
# Sync the environment
uv sync

# Run the offline tests
uv run pytest tests/

# First-time setup — credentials and user-local paths
cp .secrets.toml.example .secrets.toml             # then fill in api.username/password
cp settings.local.toml.example settings.local.toml # then set images.root

# Run Phase 0 preprocessing
uv run python scripts/run_phase0.py
```

Add a dependency:

```bash
uv add <package>
```

## Roadmap

Phased rollout per [DESIGN.md §9](DESIGN.md):

- **Phase 0** — Pull labels, build dive-level splits, fit per-dive line + wavelength priors. No model.
- **Phase 1** — Classical CV baseline (color threshold + line constraint). Establishes a floor.
- **Phase 2** — Supervised heatmap detector with presence head.
- **Phase 3** — Add line-consistency aux loss and inference-time line refinement.
- **Phase 4** — Hyperparameter sweep, failure audit, model-registry promotion.

Open operational dependency: MLflow server URL (pending).

## Layout

| File | Purpose |
| --- | --- |
| [GOAL.md](GOAL.md) | Original problem framing and observations |
| [DESIGN.md](DESIGN.md) | Architecture, training plan, evaluation, MLflow integration |
| [CLAUDE.md](CLAUDE.md) | Instructions for Claude Code working in this repo |
| `main.py` | Entry point (placeholder) |
| `pyproject.toml` | Dependencies and project metadata (managed by `uv`) |
