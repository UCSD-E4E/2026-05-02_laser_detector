# Downstream finding — bundle adjustment for IMWUT 3D reconstruction

**Where this lives**: actual analysis in `UCSD-E4E/imwut_2026_fishsense_lite`,
notebook `reconstruction_analysis/known_calibration_bundle_adjustment.ipynb`,
plus `flake.nix` for the dev environment. This note is the cross-link in
the detector repo so the conclusion is findable from `phase3_final_recipe.md`.

## TL;DR

**Don't add bundle adjustment for depth.** It doesn't help meaningfully and
the things it was meant to fix are either geometrically un-estimable, physically
negligible, or already at the irreducible Z²-amplified floor.

The detector side of this project is done. The remaining gate is upstream
(labeling) — specifically the multi-click along-line σ measurement — not more
algorithmic work.

## Setup

This work followed up on the IMWUT GitHub issue we filed:

> 2D-line-projected labels reconstruct worse than raw labels — try bundle
> adjustment over the body-frame laser ray.

The receiving instance ran the experiment in simulation (the IMWUT repo
has no real 3D ground truth, so all reconstruction work in that repo is
synthetic anyway). Geometry matches the existing `known_calibration_*`
notebooks: K with f=2850, image 4014×3016, baseline b=0.117 m, near-axial
laser direction (0,0,1).

Five label-source arms were tested:
1. Raw human labels
2. 2D self-fit ODR line projection (the deployed pipeline)
3. 2D oracle line projection (line built from noise-free endpoints)
4. BA refining only k1 distortion
5. BA refining the body-frame ray

Arms 3 and 4 from the brief (detector predictions with/without soft-snap)
were modeled as lower-σ variants of arms 2/1, not run against the real
detector — that test still requires a checkout of this detector repo +
audit parquets and is deferred.

## What we found, hypothesis by hypothesis

| candidate cause | verdict | numbers |
|---|---|---|
| **B**: closed-loop second projection re-injects fit bias | **Reproduced but tiny and geometric, not statistical** | self-fit projection: +0.85% depth RMSE vs raw; oracle projection: +0.91%. Same penalty whether or not the line was self-fit. The bias is intrinsic to perpendicular projection of Z²-weighted points. |
| **C**: un-modeled lens distortion at frame corners | **Non-issue, three independent reasons** | (1) Reprojection cost is exactly flat in k1 — k1 is un-estimable from a dive's labels under known K + free per-frame depth. (2) Laser sits at median r/f=0.059, p90=0.105 — never reaches the distorting region. (3) Even k1=−0.15 bends rays <0.1 px there vs ~2.8 px click noise. |
| **D**: per-rig ray refinement | **Degenerate under single-view geometry** | A ray with origin off by 0.4 m reprojects at the same 2.6 px noise floor as the true ray, while giving 279 m depth error. With known K + per-frame depth as a free variable, every ray fits every pixel. Refining the ray requires external 3D ground truth or per-frame pose — neither in the single-shot pipeline. |
| **E**: per-frame isotropic click noise | **Dominant, irreducible** | Depth std scales linearly with click σ (0.043 m @ 0.5 px → 0.44 m @ 5 px). No label-refinement method (2D, 3D, BA) moves this term. |
| **A** (was the original prompt's hypothesis): upstream detector line-conditioning is the problem | Not directly tested (arms 3 and 4 require real detector run) | Indirectly: B's result implies line-conditioning trades a small (~1%) depth bias for a large (~6×) cross-range gain. Direction is "small net cost to depth, big win on cross-range." |

## Why the original observation reproduces but BA can't fix it

The IMWUT observation ("line-projected labels reconstruct worse than raw") is
real, but the mechanism is geometric, not algorithmic:

- Z² triangulation weighting means far-field points dominate any line fit's
  along-line variance.
- Perpendicular projection of those points to ANY line (self-fit or oracle)
  introduces a small systematic depth offset.
- BA can re-parameterize the projection but can't escape it — without
  external constraints (distortion that's actually present, or external 3D
  GT), BA degenerates to the same projection problem.

## Bottom-line recommendation for the IMWUT paper

Don't add BA. Frame the result as a depth-error characterization, not as a
failed experiment. A clean writeup:

> We characterize the error budget of laser-triangulated depth in single-shot
> underwater imagery. Depth error decomposes into a Z²-amplified along-line
> component and a perpendicular cross-range component. Cross-range is
> recoverable via line projection (~6× reduction) at the cost of a small
> (~1%) projection-induced depth bias intrinsic to triangulation geometry,
> not to the fit method. Bundle adjustment of the body-frame laser ray is
> ineffective for depth reduction in this regime: lens distortion is
> un-estimable from labels alone, ray refinement is degenerate under
> single-view geometry, and the dominant depth term is bounded by the
> unmeasured along-line click variance.

This is a real methodological contribution — it bounds what label refinement
can buy you in this geometry, and points at the right next experiment.

## What's actually open

1. **The multi-click variance experiment** is the only instrument that yields
   the along-line click σ (the actual depth floor). Spec:
   - ~30 frames stratified by failure class
   - 3 labelers × 5 clicks per frame
   - Per-frame click σ decomposed into δ∥ / δ⊥
   - Total cost: ~1-2 hours of labeler time across 3 people
   - If δ∥ ≈ δ⊥ (say 2-3 px): we're at the depth floor; ship and move on
   - If δ∥ ≪ δ⊥ (say 0.3 px): there is more model-side gain to be had,
     potentially via sharper supervision

2. **Real-data arms 3 and 4** (detector predictions through reconstruction,
   with and without the detector's soft-snap + line-mask). Tests the
   architectural question: does line-conditioning labels upstream help or
   hurt reconstruction? B's result suggests "small net cost on depth, big win
   on cross-range" — but it's modeled, not measured. Faithful test needs the
   detector repo + run3 checkpoint + raw images.

3. **Model-as-labeler** for downstream 3D. The perpendicular-σ test I ran on
   val was inconclusive (model 0.97 px vs human 0.77 px); the proper test is
   feeding detector predictions into the reconstruction pipeline and
   comparing against human labels on the same frames. Independent of the BA
   question.

## Cross-link

`reconstruction_analysis/known_calibration_bundle_adjustment.ipynb` in
`UCSD-E4E/imwut_2026_fishsense_lite` is the actual analysis: figures, four-way
error decomposition, full simulation harness. This note is the index entry
into it from the detector repo's `notes/` folder.
