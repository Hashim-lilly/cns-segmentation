# Phase 2 — Confusion Buffer & Anki Pack (Biomedical Imaging: core + expansion + video)
### Companion to the Phase-2 daily schedule (Weeks 13–30). Copy the deck into Anki on day one; add your own cards whenever the Feynman gate exposes a gap.

**How to use with the daily plan:** the day's **Block A** concept → make a glossary line + 1–3 cards that evening → **re-derive from memory** next morning. Clear the Anki queue nightly. The **leaving bar** at the end is the real gate to Phase 3.

---

## Part 1 — Confusion Buffer (tuned for imaging)
1. **Intuition → rigour → derive/implement.** Physics on a whiteboard, losses on paper *then* in code, architectures drawn from memory.
2. **Living glossary** (nightly) + **war-chest** of one-liners for interviews.
3. **Spaced Anki:** 1 d → 3 d → 1 wk → 1 mo. Never skip a day (the queue snowballs).
4. **Feynman gate = done:** explain it aloud, no notes.
5. **Unblock ritual:** stuck >30 min → switch teacher (CS231n ↔ DigitalSreeni ↔ MONAI tutorial ↔ the paper).
6. **This phase's triangulation targets** (the concepts that most often don't click): the **Dice gradient / empty-mask instability**, **3D-conv shapes & params**, **calibration/ECE**, **MIL**, and the **SSL objective behind pathology foundation models**.
7. **Consolidation:** Weeks 16 & (implicitly) the buffer weekends — re-derive + re-implement, no new material.
8. **Understanding-gated:** don't start Phase 3 until the leaving bar is clean.

## Part 2 — Hard-topics map (ranked; where people get stuck)
1. **Loss behaviour under 95/5 imbalance** — *why* Dice/Focal/Tversky beat CE, and the empty-mask failure. (Interviews love this.)
2. **Convolution/3D-conv arithmetic** — output shapes + param counts by hand, transposed/dilated.
3. **Calibration & uncertainty** — confidence ≠ probability; MC-dropout vs ensembles; ECE.
4. **Metric choice (Metrics Reloaded)** — Dice hides boundary/small-structure errors.
5. **nnU-Net thesis** — pipeline > architecture, and *when* it fails.
6. **MIL & weak supervision** — the bag assumption for WSIs.
7. **SSL / pathology foundation models** — MAE vs DINOv2 objectives; linear-probe vs fine-tune.
8. **Domain shift & leakage** — patient-level splits; multi-vendor generalization.

## Part 3 — Anki deck (copy in; `Q → A`)

### Deck A · Imaging physics
- **Q:** What produces T1 vs T2 contrast? → **A:** T1 (longitudinal recovery) — fat/white-matter bright, fluid dark (anatomy); T2 (transverse decay) — fluid/edema bright (pathology).
- **Q:** What is FLAIR and why? → **A:** T2 with CSF signal nulled → periventricular lesions (e.g., MS) stand out against dark CSF.
- **Q:** What is T1CE? → **A:** T1 after gadolinium; enhances blood-brain-barrier breakdown (active tumor/inflammation).
- **Q:** What is the bias field; why correct it (with what)? → **A:** Smooth low-frequency intensity inhomogeneity from coil/field non-uniformity; makes identical tissue vary in intensity → breaks intensity models. Correct with **N4ITK** before training.
- **Q:** Partial volume effect? → **A:** A voxel spanning multiple tissues gets an averaged intensity → blurred boundaries and label ambiguity, worse at low resolution.
- **Q:** Three fluorescence artifacts that hurt segmentation? → **A:** Photobleaching (fading), channel bleed-through/crosstalk, autofluorescence (background emission).
- **Q:** What is k-space? → **A:** The spatial-frequency (Fourier) domain MRI is acquired in; image = inverse FFT. Center = low freq (contrast), periphery = high freq (edges).

### Deck B · Convolution math
- **Q:** Conv output-size formula (1D)? → **A:** ⌊(W − K + 2P)/S⌋ + 1.
- **Q:** Output for W=256, K=3, S=2, P=1? → **A:** ⌊(256−3+2)/2⌋+1 = 128.
- **Q:** Params in a conv layer (Cin→Cout, k×k)? → **A:** Cout·(Cin·k·k) weights + Cout biases.
- **Q:** Receptive field — what, and how does it grow? → **A:** Input region influencing one output; grows with depth/kernel/stride/dilation. Two stacked 3×3 = one 5×5 RF, fewer params, more nonlinearity.
- **Q:** Transposed conv — use + artifact? → **A:** Learnable upsampling; causes checkerboard artifacts when kernel not divisible by stride → prefer resize+conv.
- **Q:** Dilated (atrous) conv — why? → **A:** Gaps in the kernel enlarge the receptive field without extra params or resolution loss.

### Deck C · U-Net & segmentation architecture
- **Q:** Why do U-Net skip connections help? → **A:** Pass high-res spatial detail from encoder to decoder (recovering boundaries lost to downsampling) and ease gradient flow.
- **Q:** nnU-Net "no new architecture" thesis? → **A:** A plain U-Net with auto-configured preprocessing/patch-size/spacing/normalization/augmentation (from a dataset "fingerprint") beats most bespoke nets — the *pipeline* drives performance.
- **Q:** When does nnU-Net NOT win? → **A:** Very-large-context tasks, very-low-data (needs FM priors), or tasks needing a special inductive bias (e.g., topology).
- **Q:** Encoder vs decoder roles? → **A:** Encoder → semantic "what" (downsampled features); decoder → per-pixel "where" (upsampled); skips reunite them.

### Deck D · Loss functions (the imbalance killer)
- **Q:** Soft Dice loss? → **A:** 1 − (2·|P∩G| + ε)/(|P| + |G| + ε), P = probabilities, ε for stability.
- **Q:** Why is the Dice gradient unstable on empty masks? → **A:** With |G|=0 and P≈0 the ratio is dominated by ε → tiny/ill-conditioned gradients, no signal → add a CE term (DiceCE) or handle empty slices.
- **Q:** Why does CE struggle at 95/5 imbalance? → **A:** The majority class dominates the summed loss/gradient → model predicts mostly background; Dice/Focal/Tversky rebalance toward the rare class.
- **Q:** Focal loss vs CE? → **A:** Multiplies CE by (1−p_t)^γ → down-weights easy examples, focuses on hard/rare ones.
- **Q:** Tversky loss + its knob? → **A:** Generalized Dice with separate α (FP) and β (FN); raise β to punish false negatives (missing the rare structure). Focal-Tversky adds a focal exponent.
- **Q:** Boundary loss — for what? → **A:** A distance-map-weighted term penalizing boundary errors → helps thin/small structures where overlap losses are weak.
- **Q:** Why compound DiceCE? → **A:** CE = stable per-pixel gradients (early training + empty masks); Dice = direct overlap optimization (imbalance) — complementary.

### Deck E · Transformers in segmentation / foundation models
- **Q:** ViT patch embedding? → **A:** Split into fixed patches, flatten, linearly project to tokens, add positional embeddings, feed a transformer.
- **Q:** CNN vs ViT inductive biases? → **A:** CNN = locality + translation-equivariance (data-efficient, strong small-data); ViT = weak spatial prior (needs data/SSL) but captures global/long-range relations.
- **Q:** What makes SAM promptable? → **A:** Image encoded once; a light mask decoder turns point/box/mask prompts into masks in real time (zero-shot via prompting).
- **Q:** What does MedSAM change? → **A:** Fine-tunes SAM on large medical datasets (box prompt) → far better than zero-shot SAM on medical images.
- **Q:** Why does MedNeXt matter? → **A:** A ConvNeXt-style 3D U-Net that matches/beats transformer segmenters on many medical tasks — transformers aren't automatically better in low-data medical regimes.

### Deck F · 3D, uncertainty, evaluation
- **Q:** MC-Dropout for UQ? → **A:** Keep dropout ON at inference, run N passes; mean = estimate, variance/entropy = (epistemic) uncertainty.
- **Q:** Deep ensembles vs MC-dropout? → **A:** N independently-initialized models, averaged; usually better-calibrated/stronger UQ, at N× cost.
- **Q:** What is ECE? → **A:** Bin by confidence; ECE = weighted mean |accuracy − confidence| per bin. Low = probabilities match empirical accuracy (a reliability diagram shows more).
- **Q:** Metrics Reloaded message? → **A:** Pick metrics from task properties (size, boundary, prevalence), not habit; Dice rewards overlap but ignores boundaries/small structures → pair with **HD95 / NSD**.
- **Q:** What is HD95? → **A:** 95th-percentile Hausdorff distance between boundaries — boundary-sensitive, robust to a few outliers.
- **Q:** Domain shift + two fixes? → **A:** Train/test mismatch (scanner/vendor/site); fix with strong augmentation, intensity/stain normalization, domain generalization, and **site-stratified validation**.
- **Q:** 3D-conv output & params: input (D64,H128,W128,C4), k(3,3,3), s(1,2,2), p(1,1,1), 32 out? → **A:** D 64, H 64, W 64 → (64,64,64,32); params = 32·(4·27)+32 = **3,488**.

### Deck G · Phenomics & expansion (discovery-imaging)
- **Q:** What is a Cell Painting profile? → **A:** A high-dim vector of per-cell morphological features (shape/intensity/texture/granularity across organelle channels) from CellProfiler — a phenotypic fingerprint of a perturbation.
- **Q:** Phenomics dose-response readout? → **A:** Signal vs concentration → fit a sigmoid for EC50/IC50; ROC-AUC for hit-calling — *after* batch correction.
- **Q:** Batch effects — what & why care? → **A:** Non-biological variation (plate/day/edge-well/session) that can swamp biology; correct (e.g., normalize to negative controls) and validate batch-aware.
- **Q:** Cellpose vs StarDist? → **A:** Cellpose = gradient-flow to cell centers (arbitrary shapes, generalist); StarDist = star-convex polygons (great for round nuclei). Both instance segmentation.
- **Q:** Registration types? → **A:** Rigid (rot+trans), affine (+scale/shear, linear), deformable (per-voxel nonlinear warp). More flexibility fits better but risks unrealistic warps.
- **Q:** How does registration error hit region readouts? → **A:** Misalignment mislabels boundary voxels into wrong atlas regions → biased per-region metrics; report registration QC (Dice-to-atlas) alongside.
- **Q:** Why can't you load a WSI normally; fix? → **A:** Gigapixel (20–50 GB) > RAM; use pyramidal/tiled formats (OpenSlide/zarr) + dask, tile with overlap/context, stitch.
- **Q:** What is MPP and why care? → **A:** Microns-per-pixel (resolution/magnification); apply models at their training MPP or resample — mismatch silently degrades accuracy.
- **Q:** What does stain normalization fix? → **A:** Lab/scanner H&E colour variation; maps stain colour to a reference via stain-vector decomposition (Macenko/Vahadane) → cross-site robustness.
- **Q:** The MIL assumption? → **A:** A bag (slide) is positive iff ≥1 instance (tile) is positive → learn from bag labels without tile annotation (weak WSI labels).
- **Q:** What does active learning optimize? → **A:** Which unlabeled samples to annotate next (uncertainty/diversity) to maximize gain per label → less annotation.
- **Q:** SSL objective of MAE vs DINOv2? → **A:** MAE = masked image modeling (reconstruct masked patches); DINOv2 = self-distillation (student matches teacher features across augmentations). Both label-free.
- **Q:** When do pathology-FM embeddings beat a supervised CNN? → **A:** Low-label regimes + cross-cohort transfer; a linear probe on FM embeddings often beats a from-scratch CNN. Fine-tune when labels are ample and the domain gap is large.

### Deck H · Video / temporal (2V)
- **Q:** Tracking-by-detection vs joint? → **A:** TBD = detect per frame then associate (ByteTrack); joint = learn detection+association together. TBD modular/strong; joint fuses motion+appearance.
- **Q:** Why is ByteTrack effective? → **A:** It associates high-AND low-confidence detections (low-score boxes are often occluded objects) → fewer ID switches/misses.
- **Q:** What is HOTA? → **A:** Higher-Order Tracking Accuracy — balances detection and association in one score (unlike MOTA which over-weights detection).
- **Q:** How does SAM 2 track across video? → **A:** A streaming memory of past frames' features/masks; it attends to memory to propagate/segment in new frames.
- **Q:** What does RAFT estimate? → **A:** Per-pixel optical flow (motion field) via iterative refinement over a 4D correlation volume.
- **Q:** DeepLabCut/SLEAP do what? → **A:** Markerless animal pose estimation (track user-defined keypoints via transfer-learned CNNs) → feed behavior classification.

## Part 4 — Common misconceptions & traps (high-value)
- **"Higher Dice = better model."** No — Dice ignores boundaries and small structures; a model can win on Dice yet fail clinically. Pair with HD95/NSD.
- **"A fancier architecture beats nnU-Net."** Usually not; the auto-configured *pipeline* dominates. Beating it needs data-/task-specific reasons.
- **"Softmax/sigmoid confidence is a probability."** Only if calibrated — check ECE/reliability; apply temperature scaling.
- **"SAM segments medical images zero-shot."** Poorly; use MedSAM or fine-tune.
- **"Same-site train/test generalizes."** Domain shift (scanner/vendor/site) breaks it — validate across sites, split at the **patient** level (no patient in both train and test) to avoid leakage.
- **"Instance = semantic segmentation."** No — instance separates individual objects (each cell), semantic labels classes.
- **"Bigger input/patch always helps 3D."** Memory-bound; patch size, spacing, and batch interact — nnU-Net *fingerprints* these for a reason.

## Part 5 — Glossary starter (seed; expand nightly)
Dice / IoU · Focal / Tversky / boundary loss · compound DiceCE · receptive field · transposed / dilated conv · skip connection · nnU-Net fingerprint · patch embedding · promptable segmentation (SAM/MedSAM) · MC-dropout · deep ensemble · ECE / reliability diagram · HD95 / NSD · domain shift · bias field / N4 · partial volume · k-space · Cell Painting · EC50/IC50 · batch effect · instance vs semantic segmentation · MPP · tiling / zarr / pyramidal · stain normalization (Macenko/Vahadane) · MIL (bag/instance) · active learning · SSL (MAE / DINOv2) · foundation model / linear-probe vs fine-tune · tracking-by-detection · HOTA · optical flow · pose estimation.

## Part 6 — Drills
**Whiteboard (no notes):** conv output-size + param count; the 3D-conv drill above; derive the Dice gradient + explain empty-mask instability; draw U-Net with per-layer shapes for 256×256; explain ECE and why confidence≠probability; state the MIL assumption; contrast CNN vs ViT inductive biases.
**Blank-file (no AI):** Dice / Focal / Tversky / boundary / DiceCE + a gradient-vs-class-ratio plot; conv2d from scratch; MC-dropout + predictive entropy; a tile→infer→stitch loop for a large image; a minimal Cellpose/StarDist run + one morphological feature.

## Part 7 — Triangulation (for the topics that don't click)
- **Losses / Dice gradient:** the loss survey (Jadon, arXiv 2006.14822) + implement + plot gradients yourself.
- **Conv arithmetic:** Dumoulin & Visin (arXiv 1603.07285) + the animation repo.
- **nnU-Net:** read the paper twice *and* the v2 source (github.com/MIC-DKFZ/nnUNet).
- **Metrics/UQ:** Metrics Reloaded (Nature Methods 2023) + Yarin Gal's UQ lectures.
- **Phenomics:** CellProfiler + the JUMP-CP protocol; RxRx docs.
- **Pathology FMs / SSL:** the Virchow (Nature Medicine 2024) paper + DINOv2/MAE repos + UNI on HF.
- **Tracking:** ByteTrack + the HOTA paper (arXiv 2009.07736).

## Part 8 — Spaced-review & the leaving bar
**Daily** Anki + one re-derivation; **weekly** Whiteboard-Friday + a teach-a-junior note; trust the 1/3/7/30-day schedule; **Week 16** consolidation cements the shakiest items.
**Leaving bar (cold, unaided) →** compute conv/3D-conv shapes & params; derive the Dice gradient + empty-mask instability; draw U-Net + state the nnU-Net thesis and when it fails; pick metrics for a given task and justify; explain MC-dropout + ECE (confidence≠probability); describe a Cell-Painting profile + batch effects; state the MIL assumption; name the SSL objective behind a pathology FM and when its embeddings beat a CNN; contrast tracking-by-detection vs joint + define HOTA.
