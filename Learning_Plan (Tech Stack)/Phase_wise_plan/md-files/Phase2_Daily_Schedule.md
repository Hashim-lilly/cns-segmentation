# Phase 2 — Day-by-Day Schedule · Biomedical Imaging (core + expansion + video)
### Weeks 13–30 · Mon Nov 30, 2026 → Sun Apr 4, 2027 · ~385 hrs (+ Capstones 1 & 2) · → Tier 0/1

**Goal:** own MRI / fluorescence / whole-slide segmentation end-to-end, then extend into phenomics, atlas registration, WSI engineering, and pathology foundation models; finish with video/temporal. Two shippable capstones.

**Blocks:** **A** 06:00–08:00 (theory) · **B** 08:30–10:30 (build + threads) · **Evening** 20:30–22:30 (papers, Anki, R). **Weekend** = ~10 h buffer + one rest day.
**Threads (running):** **T** DSA+C++ (first ~45 min of Block B) · **M** implement-from-scratch (Block B) · **R** reproduce-a-paper + write + apply (Evening, ~2 h/wk).

---

## Block 2A — Segmentation core (Weeks 13–22)

### Week 13 · Nov 30–Dec 6 — Imaging physics
| Day | Block A | Block B (T/M + build) | Evening (+ R) |
|---|---|---|---|
| Mon | MRI contrast: T1/T2/FLAIR/T1CE (MRI Q&A) | **T:** trees/BFS-DFS → load & view public MRI (nibabel) | OHBM MRI physics; Anki |
| Tue | MRI artifacts: motion, bias field, partial volume | **M:** bias-field correction demo (N4/SITK) | Callaghan L1–2; **R:** pick imaging paper |
| Wed | Fluorescence: photobleaching, bleed-through, autofluorescence (iBiology) | **T:** heaps → inspect a fluorescence stack | Anki; glossary |
| Thu | k-space & acquisition basics | **M:** simple intensity normalization pipeline | Callaghan L3–4 |
| Fri | Preprocessing decisions that affect model design | **T:** intervals/greedy | Whiteboard-Fri: which artifacts to correct & why; **R:** write-up |
| Sat–Sun | **Buffer + rest** · *Deliverable:* 1-page physics note (T1/T2/FLAIR/T1CE + fluorescence artifacts) | | |

### Week 14 · Dec 7–13 — CNNs, U-Net family, convolution math
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | CS231n L5: CNNs; conv arithmetic (Dumoulin) | **T:** DP intro → implement conv2d from scratch | Anki |
| Tue | U-Net (read; derive output sizes) | **M:** hand-compute U-Net shapes; code encoder | Re-read U-Net; Anki |
| Wed | V-Net (3D) + Attention U-Net | **T:** DP on grids → code decoder + skips | Anki |
| Thu | nnU-Net paper (read twice) | **M:** finish a minimal U-Net; overfit 1 image | nnU-Net paper again; **R:** reproduce-paper |
| Fri | Receptive fields; why skips work | **T:** graphs/union-find | Whiteboard-Fri: U-Net from memory + shapes |
| Sat–Sun | **Buffer + rest** | | |

### Week 15 · Dec 14–20 — Loss functions (the imbalance killer)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Loss survey (Jadon); Dice math | **M:** Dice + BCE from scratch; unit-test | Anki |
| Tue | Generalized Dice; class weighting | **M:** Generalized Dice; gradient vs class-ratio plot | Anki |
| Wed | Focal & Tversky (severe imbalance) | **M:** Focal, Tversky, Focal-Tversky | Anki |
| Thu | Boundary loss (thin structures) | **M:** Boundary loss + compound DiceCE | **R:** write-up |
| Fri | When each loss helps | **T:** review + timed set | Whiteboard-Fri: derive Dice gradient + empty-mask instability |
| Sat–Sun | **Buffer + rest** · *Deliverable:* tested losses library + gradient plot | | |

### Week 16 · Dec 21–27 — 🛑 CONSOLIDATION (holiday-light, ~18 h)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon–Fri | Re-derive: conv output sizes, Dice gradient; re-read nnU-Net design | Re-implement (blank file): conv2d + Dice; **T:** spaced review | Anki catch-up; **R:** finish a reproduce-paper |
| Sat–Sun | **Rest** (holidays) | | |

### Week 17 · Dec 28–Jan 3 — nnU-Net deep dive + MONAI
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | nnU-Net fingerprinting → auto-config | **T:** DP/backtracking → clone & run MONAI brain-seg notebook | MONAI videos; Anki |
| Tue | MONAI transforms & data pipeline | **M:** build a MONAI training loop | Anki |
| Wed | nnU-Net v2 source read | **T:** matrix/2D problems → modify nnU-Net config | Kitware comparison |
| Thu | Patch-based training, sliding-window inference | **M:** sliding-window inference | **R:** reproduce-paper |
| Fri | Preprocessing/spacing/normalization choices | **T:** timed medium set | Whiteboard-Fri: walk nnU-Net config end-to-end |
| Sat–Sun | **Buffer + rest** | | |

### Week 18 · Jan 4–10 — MONAI + microscopy; reproduce nnU-Net
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | AI-for-Medical-Diagnosis: brain-MRI U-Net module | **T:** strings → set up BraTS-subset run | DigitalSreeni (U-Net) |
| Tue | Evaluation module (metrics) | **M:** train nnU-Net on BraTS subset | Anki |
| Wed | Microscopy-native tips (DigitalSreeni) | **T:** two-pointer/window → adapt to fluorescence | DigitalSreeni code-along |
| Thu | What changes MRI→fluorescence | **M:** fluorescence tiling + inference | **R:** write-up |
| Fri | Review core seg | **T:** timed set | Whiteboard-Fri: nnU-Net defaults, what you'd change for microscopy |
| Sat–Sun | **Buffer + rest** · *Deliverable:* reproduced nnU-Net + write-up | | |

### Week 19 · Jan 11–17 — Transformer segmentation + foundation models
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | ViT (re-derive attention) | **M:** patch embedding + a ViT block | Anki |
| Tue | TransUNet / Swin-UNet | **T:** graphs → run a Swin-UNet demo | Anki |
| Wed | UNETR / Swin UNETR (3D) | **M:** wire a UNETR-style decoder | Anki |
| Thu | SAM / MedSAM (promptable) | **T:** intervals → run MedSAM on a sample | **R:** reproduce-paper |
| Fri | MedNeXt (ConvNet that still wins) | **T:** timed set | Whiteboard-Fri: CNN vs ViT inductive biases; SAM design |
| Sat–Sun | **Buffer + rest** | | |

### Week 20 · Jan 18–24 — 3D, uncertainty, evaluation
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | TorchIO: volumetric augmentation | **M:** a 3D augmentation pipeline | Anki |
| Tue | MC-Dropout, deep ensembles (Yarin Gal) | **M:** add MC-Dropout + predictive entropy | Anki |
| Wed | Calibration (ECE, reliability) | **T:** DP → compute ECE + reliability diagram | Anki |
| Thu | Metrics Reloaded (choosing metrics) | **M:** proper Dice/IoU/HD95 eval | **R:** write-up |
| Fri | Domain shift (Stanford AIMI) | **T:** 3D-conv shape/param drill (by hand) | Whiteboard-Fri: pick metrics for a task; UQ readout |
| Sat–Sun | **Buffer + rest** | | |

### Weeks 21–22 · Jan 25 – Feb 7 — 🎯 CAPSTONE 1 (public brain-seg) + self-test
| Day (both weeks) | Block A | Block B | Evening (+ R) |
|---|---|---|---|
| Mon–Fri (Wk 21) | Design decisions (preprocessing→arch→loss→UQ) | Build: preprocessing + 3-architecture ablation on BraTS/Decathlon subset | Log results; **R:** capstone as public repo |
| Mon–Thu (Wk 22) | Analysis & failure modes | Loss ablation + UQ + failure-mode analysis; **T:** keep 1 timed set/day | Write the 3-page write-up |
| Fri (Wk 22) | **🚩 PHASE-2-CORE SELF-TEST** (U-Net shapes; Dice gradient; why nnU-Net wins; diagnose bias-field/imbalance/label-noise/domain-shift) | Polish repo + README | **R:** post write-up; **▶ Tier 0/1 applications** |
| Sat–Sun | **Buffer + rest** · *Deliverable:* Capstone-1 notebook + write-up (public) | | |

## Block 2B — Discovery-imaging expansion (Weeks 23–26)

### Week 23 · Feb 8–14 — Cell/nucleus segmentation + phenomics (2.6)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Cellpose & StarDist (instance seg) | **M:** run Cellpose + StarDist on a public cell set | Anki |
| Tue | Cell Painting assay + morphological features | **T:** graphs → CellProfiler feature extraction | JUMP-CP / RxRx overview |
| Wed | Dose–response / EC50, ROC-AUC readouts | **M:** fit a dose–response curve | Anki |
| Thu | Batch effects in phenomics | **T:** DP → batch-correction demo | **R:** reproduce-paper |
| Fri | Profiling pipeline design | **T:** timed set | Whiteboard-Fri: what a Cell-Painting profile encodes |
| Sat–Sun | **Buffer + rest** · *Deliverable:* segment → features → AUC readout (public) | | |

### Week 24 · Feb 15–21 — 3D + atlas registration (2.7) + WSI engineering start (2.8)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Registration: rigid/affine/deformable | **M:** ANTsPy register a mouse-brain volume | Anki |
| Tue | Allen atlas; BrainGlobe/brainreg | **T:** trees → atlas-based region readout | Anki |
| Wed | Registration error → metric propagation | **M:** per-region quantification | Anki |
| Thu | WSI: tiling, pyramidal I/O, zarr/dask | **T:** strings → OpenSlide tiling script | TIAToolbox docs; **R:** write-up |
| Fri | Keeping tile context; throughput | **T:** timed set | Whiteboard-Fri: registration types + error propagation |
| Sat–Sun | **Buffer + rest** | | |

### Week 25 · Feb 22–28 — WSI finish (2.8) + stain-norm/MIL/active learning (2.9)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Tiled inference + stitching (TIAToolbox) | **M:** tiled inference on a public WSI | Anki |
| Tue | Stain normalization (Macenko/Vahadane) | **M:** torchstain normalization + ablation | Anki |
| Wed | MIL for weak slide labels (CLAM) | **T:** graphs → CLAM MIL classifier | Anki |
| Thu | Active learning (uncertainty/diversity) | **M:** an active-learning loop (modAL) | **R:** reproduce-paper |
| Fri | Annotation efficiency | **T:** timed set | Whiteboard-Fri: MIL assumption + stain-norm rationale |
| Sat–Sun | **Buffer + rest** | | |

### Week 26 · Mar 1–7 — Pathology foundation models & SSL (2.10) + imaging-track capstone
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | DINOv2 / MAE (SSL objectives) | **M:** extract UNI embeddings on a public task | Virchow paper (skim) |
| Tue | Pathology FMs (UNI/CONCH/Virchow) | **T:** DP → linear-probe on FM embeddings | Anki |
| Wed | Linear-probe vs fine-tune | **M:** fine-tune head; compare to supervised CNN | Anki |
| Thu | When FM embeddings win (low-data) | **T:** timed → build the comparison | **R:** write-up |
| Fri | Wrap the imaging-track capstone | Polish repo (seg → features → FM-embeddings comparison) | Whiteboard-Fri: SSL objective; FM vs CNN |
| Sat–Sun | **Buffer + rest** · *Deliverable:* imaging-track capstone (public) — MIDL/MICCAI-workshop candidate | | |

## Block 2C — Imaging & video extension (2V) (Weeks 27–30)

### Week 27 · Mar 8–14 — Video & temporal models
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | VideoMAE / TimeSformer | **M:** a temporal model on a toy clip set | Anki |
| Tue | ViViT / SlowFast | **T:** DP-on-sequences → run a clip classifier | Anki |
| Wed | SAM 2 (memory for video seg) | **M:** SAM 2 mask propagation on a clip | Anki |
| Thu | RAFT (optical flow) | **T:** graphs → optical-flow demo | **R:** reproduce-paper |
| Fri | Video-transformer vs frame+track | **T:** timed set | Whiteboard-Fri: SAM 2 memory mechanism |
| Sat–Sun | **Buffer + rest** | | |

### Week 28 · Mar 15–21 — Tracking + pose
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Tracking-by-detection; ByteTrack | **M:** run ByteTrack on a public clip | Anki |
| Tue | OC-SORT / BoT-SORT; HOTA metric | **T:** intervals → compute HOTA | Anki |
| Wed | Pose: DeepLabCut | **M:** DeepLabCut on a sample | Anki |
| Thu | Pose: SLEAP | **T:** graphs → SLEAP run | **R:** write-up |
| Fri | ID-switching & re-ID | **T:** timed set | Whiteboard-Fri: tracking-by-detection vs joint |
| Sat–Sun | **Buffer + rest** | | |

### Weeks 29–30 · Mar 22 – Apr 4 — Cell tracking/lineage + 🎯 CAPSTONE 2 + self-test
| Day | Block A | Block B | Evening (+ R) |
|---|---|---|---|
| Mon–Fri (Wk 29) | Cellpose→TrackMate/Ultrack; lineage | Build Capstone 2 (pose→behavior **or** cell-track→lineage); SAM 2 for masks; **T:** 1 timed set/day | Log; **R:** capstone repo |
| Mon–Thu (Wk 30) | Lineage failure modes; eval | Finish + evaluate + write-up; **T:** keep timed sets | Write-up |
| Fri (Wk 30) | **🚩 PHASE-2V SELF-TEST** (SAM 2 memory; tracking-by-detection vs joint; when a video transformer wins; lineage failure modes) | Polish repo + README | **R:** post; **▶ Tier 1 applications** (Recursion/PathAI) |
| Sat–Sun | **Buffer + rest** · *Deliverable:* Capstone 2 (public) | | |

**End of Phase 2 → Phase 3 (DL theory) next.** You now hold Tier 0/1: segmentation + phenomics + video, two public capstones, threads live.
