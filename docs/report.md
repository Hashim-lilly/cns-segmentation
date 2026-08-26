# Technical Report — Multi-Structure Spinal Segmentation Pipeline

**Status:** Phase 5 deliverable. All numbers below are read directly from real output files
(`experiments/evaluate_report/merged_metrics.csv`, `experiments/baselines_report/merged_metrics.csv`,
`experiments/_phase5_dashboard_verify/verification_report.json`) produced by real GPU/CPU runs — none
are estimated or backfilled. See `ROADMAP.md` for the full phase-by-phase history this report draws on;
where the two disagree, ROADMAP.md is authoritative.

---

## 1. Pipeline overview

T2-weighted spine MRI → SegResNet (3D) multi-class segmentation → MC-Dropout uncertainty → mesh
export → per-case metrics, exposed through a CLI (`scripts/predict.py`, `scripts/evaluate.py`,
`scripts/evaluate_external.py`, `scripts/uncertainty.py`, `scripts/run_baselines.py`) and an
interactive Streamlit dashboard (`src/cns_segmentation/demo/app.py`).

Four structures are modeled, each its own checkpoint (per-structure training, Phase 1):

| Structure | Checkpoint val Dice (training-time) | Classes in checkpoint |
|---|---|---|
| cord | 0.9507 | background, cord |
| canal | 0.9501 (ROADMAP Phase 1) / 0.9461 (this eval, MC-Dropout mean, see §5) | background, canal, cord |
| csf | 0.5512 | background, canal, csf, cord |
| rootlets | 0.6121 | background, canal, cord, rootlets |

Evaluation metrics (Dice, Hausdorff95, volume error, Normalized Surface Dice) come from
`evaluation/metrics.py`'s `evaluate_subject()`/`aggregate_metrics()`, unmodified in this phase.
Calibration (ECE, reliability diagrams) comes from Phase 3's `evaluation/calibration.py`.

---

## 2. Held-out and external evaluation results

Source: `experiments/evaluate_report/merged_metrics.csv`, produced by `scripts/evaluate.py` reusing
Phase 3's MC-Dropout run artifacts (`experiments/uncertainty_<structure>/`) and Phase 2's external
cross-dataset runs (`experiments/external_eval_<dataset>/`), re-aggregated via `aggregate_metrics()`.

### 2.1 Held-out, same-distribution (spine-generic sites not seen in training)

| Structure | N | Dice | HD95 (mm) | Volume error (mm³) | Surface Dice |
|---|---|---|---|---|---|
| cord | 32 | 0.9507 ± 0.0095 | 0.68 ± 0.19 | 180 ± 134 | 0.9953 ± 0.0053 |
| canal | 50 | 0.9461 ± 0.0104 | 0.67 ± 0.17 | 387 ± 307 | 0.9944 ± 0.0057 |
| csf | 2 | 0.7964 ± 0.0226 | 15.4 ± 13.1 | 2404 ± 1748 | 0.8904 ± 0.0320 |
| rootlets | 4 | **0.0000 ± 0.0000** | undefined (n=0 finite) | 835 ± 159 | **0.0000 ± 0.0000** |

Cord and canal are strong and consistent across every site (per-site Dice range 0.943–0.953 for cord,
0.933–0.951 for canal — see the full CSV for per-site breakdowns). csf's n=2 is too small to draw a
generalization conclusion from. **Rootlets scores Dice=0.0 on all 4 true held-out subjects** — see §5,
this is a genuine, previously-diagnosed (Phase 3) argmax-collapse failure, not a metrics bug.

### 2.2 External cross-dataset validation (different acquisition source entirely)

| Structure | External dataset | N | Dice | HD95 (mm) | Notes |
|---|---|---|---|---|---|
| canal | spider_canal (lumbar, 1 source) | 210 | 0.0905 ± 0.0943 | 153.2 ± 46.2 (n=164) | Checkpoint trained cervical-thoracic only; never saw lumbar canal anatomy (Phase 2 root cause) |
| rootlets | openneuro_ds004507 | 7 | 0.0000 ± 0.0000 | undefined | Max softmax for rootlets channel was 0.1018 volume-wide (Phase 2 debug pass) — never wins argmax |

Both external failures are real, diagnosed domain-shift/generalization gaps (region mismatch for
canal, acquisition-protocol mismatch for rootlets), documented in ROADMAP.md Phase 2 with the exact
debug evidence. They are not artifacts of this evaluation pass.

---

## 3. Comparison baselines

Source: `experiments/baselines_report/merged_metrics.csv` + `blockers.yaml`, produced by
`scripts/run_baselines.py` against the same 32 held-out cord subjects (5 sites) and 4 held-out
rootlets subjects (2 sites) used in §2.1, so the comparison is apples-to-apples.

### 3.1 SCT DeepSeg (real, full sweep, no blockers)

| Structure | Baseline | N | Dice | HD95 (mm) | Volume error (mm³) |
|---|---|---|---|---|---|
| cord | `sct_deepseg spinalcord` | 32 | **0.9857** ± 0.0079 | 0.75 ± 0.19 | 131 ± 87 |
| rootlets | `sct_deepseg rootlets` | 4 | **0.6909** ± 0.0192 | 5.58 ± 1.19 | 100 ± 61 |

### 3.2 TotalSegmentator-MRI — documented blocker, not fabricated

`total_mr --fast -d cpu` (no GPU build available in this environment) was run against the same held-out
cord subjects. It timed out after the script's 900s per-subject limit on the very first subject
(`sub-stanford01`), even in `--fast` mode on CPU. Per the plan's explicit instruction to document
failures rather than invent numbers, this baseline reports **zero scored subjects** and the blocker
`totalsegmentator_cord: totalsegmentator_failed` (`experiments/baselines_report/blockers.yaml`). No
TotalSegmentator-MRI numbers appear anywhere in this report.

### 3.3 Head-to-head: this project's model vs. SCT DeepSeg, same subjects

| Structure | This model (Dice) | SCT DeepSeg (Dice) | Gap |
|---|---|---|---|
| cord | 0.9507 | 0.9857 | SCT +0.035 |
| rootlets (true held-out) | **0.0000** | 0.6909 | SCT +0.69 |

For cord, SCT's pretrained model modestly outperforms this project's in-house cord checkpoint on the
same 32 subjects — both are strong, production-viable numbers. For rootlets, the gap is not marginal:
SCT's pretrained rootlet model produces real, non-trivial predictions on the same subjects where this
project's own rootlets checkpoint predicts no rootlet voxels at all. This is the report's most
important negative finding and is stated plainly rather than being buried in a caveat: **the in-house
rootlets model does not currently generalize to held-out sites, despite a respectable 0.6121
training-time validation Dice (§5)**, and a pretrained external tool already does better on the exact
same data.

---

## 4. Dashboard walkthrough

`src/cns_segmentation/demo/app.py` implements all 7 planned features. Verified two ways (no browser
automation available in this environment, confirmed with the user in advance):

1. **Real headless server**: `streamlit run app.py --server.headless true --server.port 8511`, then
   `curl` against both `/` and `/_stcore/health` — both returned real HTTP 200 (genuine Streamlit HTML
   and `ok`, not just a process-alive check).
2. **Real feature-by-feature execution** against a real held-out subject (`sub-barcelona01`, structure
   `canal`), saving concrete artifacts. Full results in
   `experiments/_phase5_dashboard_verify/verification_report.json`:

| # | Feature | Verified result |
|---|---|---|
| 1 | Upload/select | 50 real subjects listed via `dataset_registry`; picked `sub-barcelona01` (site: barcelona) |
| 2 | Segment + progress | Real inference, output shape (51, 511, 511) matches input; predicted classes {0,1,2}; uncertainty maps present |
| 3 | 3D slice render | Real Plotly figure, 2 traces (image + prediction overlay) |
| 4 | Uncertainty overlay | Real Plotly figure, 3 traces (image + prediction + uncertainty layer) |
| 5 | Mesh preview | Real non-empty mesh figure (marching-cubes only, no repair — explicitly labeled "preview only" in the UI, not a CFD-readiness claim) |
| 6 | Export | Real NIfTI ×4 (prediction, entropy, variance, mutual information) + real STL written to disk |
| 7 | Per-case metrics | Real Dice=0.8072, HD95=3.16mm, volume error=11476mm³, surface Dice=0.8019, ECE=0.0035, 15-bin reliability diagram (144KB PNG) |

**Feature 6's exported mesh is explicitly not CFD-ready**, and the dashboard shows this rather than
hiding it: `MeshQuality` for this export is `watertight=False, manifold=False, euler_number=-7,
passes_cfd_check=False`. This matches Phase 4's finding that real-subject meshes fail full CFD intake
without Phase 1's joint multi-class model or heavier repair — the dashboard surfaces the same
pass/fail flag next to the download button rather than implying the STL is simulation-ready.

**A real bug was found and fixed during this verification**, not assumed away: Feature 7's original
implementation compared the dashboard's resampled prediction grid against a raw, un-resampled
ground-truth label and crashed with a shape mismatch (`(51, 511, 511)` vs `(64, 320, 320)`). Fixed by
adding a label-only resample step (`_resample_label_to_grid()`) using the same
`Orientationd`(RAS)+`Spacingd`(nearest-neighbor) transform the batch scripts already use for image+label
jointly — scoped to the dashboard's interactive code path only; `evaluation/metrics.py` was not
modified. Full test suite (198 tests) passes after the fix, confirming no regression.

---

## 5. Limitations

Stated plainly, per the project's standing "verify, don't fabricate" requirement:

1. **Rootlets segmentation does not generalize.** True held-out Dice is 0.0 on 4 subjects and 0.0 on 7
   external subjects, despite a 0.6121 training-time validation Dice. Phase 3 root-caused this as an
   argmax-collapse: the rootlets channel's own native validation split shows the same collapse under
   full-image sliding-window inference (max softmax ~0.10, never wins argmax), even with MC-Dropout off
   and `keep_largest_component` off — ruling out those as causes. **The training-time validation Dice
   and the full-image inference Dice measure different things and must not be conflated** — this is a
   real, unresolved reproducibility gap between training-time patch-based validation and full-volume
   inference, not a bug introduced by this report. SCT's pretrained rootlet baseline (Dice 0.69 on the
   same subjects) shows the anatomical signal is learnable; this project's own checkpoint isn't
   currently capturing it at inference time.
2. **csf's held-out set is n=2; rootlets' external+held-out sets are n≤7.** Neither is large enough
   for a statistically meaningful generalization estimate. Numbers are reported because they're real,
   not because they're conclusive.
3. **Held-out numbers for canal/cord/csf/rootlets are MC-Dropout mean-probability outputs** (Phase 3's
   artifacts, reused by `scripts/evaluate.py` rather than re-run), not a fresh deterministic
   `predict.py` argmax pass. They are close to — but not guaranteed bit-identical to — a deterministic
   rerun. Canal's Dice here (0.9461) vs. ROADMAP's Phase 1 deterministic figure (0.9501) illustrates the
   size of this gap: real, but small.
4. **TotalSegmentator-MRI produced zero baseline numbers.** It timed out after 900s per subject even in
   `--fast` CPU mode; no GPU build was available in this environment. This is a documented
   infrastructure blocker, not a claim that TotalSegmentator cannot segment cord.
5. **Mesh preview ≠ CFD-ready.** The dashboard's STL export uses fast marching-cubes with no repair
   pass, by design (Deliverable 3 spec) — it is a visualization convenience, not validated mesh
   geometry. Even the *fully repaired* mesh pipeline (Phase 4 Track A) still fails the CFD-readiness
   gate on 0/3 real spine-generic subjects tested (Euler numbers −184/−224/−90, all non-2) — durably
   fixing this needs Phase 1's joint multi-class model (mutually-exclusive labels by construction, one
   network, no cross-model boundary disagreement), not more mesh-repair tooling.
6. **Canal generalizes within its trained spinal region only.** Dice collapses from 0.9461 (held-out
   cervical-thoracic) to 0.0905 (external, lumbar) — a real cross-region domain-shift failure, not an
   external-dataset quality issue (Phase 2 root cause: SPIDER's images are anatomically outside what
   the canal checkpoint was ever trained on).
7. **No independent test set for the dashboard's headless verification** — features were verified
   against one real subject (`sub-barcelona01`), not a systematic sweep. This proves the 7 features
   execute correctly end-to-end against real data; it is not a claim of dashboard-wide statistical
   coverage.
