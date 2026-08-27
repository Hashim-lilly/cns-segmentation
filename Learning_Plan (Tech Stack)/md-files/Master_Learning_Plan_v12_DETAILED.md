# Drug-Discovery + Biomedical-Imaging ML — Research-Engineer Readiness — Master Learning Plan (v12 · DETAILED edition)

**Owner:** MD Hashim · **Start:** Sep 1, 2026 (Week 0 = bio-clock + new-project ramp; plan proper from Sep 8) · **Commitment:** **~40–42 h/week** = **30 h structured weekday** (Block A 06:00–08:00 · Block B 08:30–10:30 · Evening 20:30–22:30, ×5) **+ ~10 h weekend buffer** · **Total:** ~3,000 hrs · **Full-depth horizon:** ~mid-2028, laddered readiness across 2027 · **Targets:** ML Research Engineer / Applied Scientist at DeepMind-science, Isomorphic Labs, D. E. Shaw Research, Recursion/Insitro, top biomedical-imaging AI — and your organization's small-molecule & discovery-imaging AI work.

> **This is the DETAILED edition of v12.** Every phase is elongated to the v2 pattern — week-by-week, with an **hour budget + how-to-use note + inline URL on every resource**, concrete **Deliverables**, **Whiteboard checkpoints**, per-phase **capstones**, and a phase **interview self-test**. Threads are expanded to the same depth, and the resource index is fused into one topic-organized list. Every squad-derived skill is rebuilt with **public methods + public datasets + personally-built capstones**; no internal codenames/targets/Jira-IDs/teammate or partner names appear here or in any resume/public/interview artifact.

---

## 1. Weekly hours, timeline & a real-life feasibility check (read first)

| | Hours | Content |
|---|---|---|
| **Weekday structured** (Mon–Fri) | **30 h** (6 h/day: Block A 2h + Block B 2h + Evening 2h) | new learning per the phase |
| **Weekend buffer** (Sat–Sun) | **~10 h** (2 × ~5 h) | revision, weekday spillover, emergencies — *not* new material by default |
| **Total** | **~40–42 h/week** | |

**Effort by block (cross-checked to ~2,925 h):** Threads T+M+R ~550 (concurrent) · P0 55 · P1 210 · P2 core+Capstone1 220 · P2 imaging expansion 165 · P2V+Capstone2 100 · P3 165 · P4 165 · **Molecular-ML Track 5A–5F 450** · Phase Q+CapstoneQ 335 · P6+Capstone4 175 · P7+Capstone5 175 · P8 160.

**Timeline (sequential spine + Phase Q at ~30 h/wk; threads run concurrently, so they add no calendar time):** ~**87 calendar weeks ≈ ~20 months → full depth ~mid-2028** — deliberately unhurried; see §7 for the week-by-week schedule. **Job-readiness is laddered far earlier** (apply from Week 3; Tier 2 by ~mid-Oct 2027; Tier 3 crescendo ~mid-Feb 2028).

**Feasibility — honest verdict.** 6 h of focused learning on top of a ~8.5 h office day is ~14–15 h of cognitive work/day — genuinely at the upper edge of sustainable. Guardrails are **load-bearing**: protect ~7 h sleep (lights-out ~22:45, wake ~05:45 — it's when the morning's learning consolidates); the **Evening block flexes first** when tired (push to weekend); one full rest day; consolidation weeks every ~5 weeks; plan 30 h, expect real delivery ~26–30 h and let the buffer absorb the gap; act on burnout signals early.

**Parallel new project (critic agentic doc-review pipeline).** Real GenAI/agentic work, directly on the Phase-7 path (critique/verify loops, MCP). Treat as office work, let it double as Phase-7 experience, keep its specifics off public artifacts, and don't let it eat Block A/B learning time.

---

## 2. The daily rhythm — cognitive-focus mapping

| Window | State | Do this | Concretely |
|---|---|---|---|
| **Block A · 06:00–08:00** | **Peak** | **Hardest NEW theory + derivation**, intuition-first (visual → rigorous → derive by hand). Never email/video-binge. | New math; DL theory; GNN/GFlowNet/FEP/UQ/quantum theory. |
| **Block B · 08:30–10:30** | **High, warmed-up** | **Active building & problem-solving.** Turn Block-A theory into code. | Thread M (implement-from-scratch); capstone coding; Thread T DSA. |
| **Evening · 20:30–22:30** | **Lower (post-work)** | **Review, intake, consolidation.** | Lecture/seminar videos; paper reading (Thread R); Anki + glossary; re-do solved DSA; plan tomorrow's Block A. |

**Daily spacing loop:** learn (A) → build (B) → review (Evening) → re-derive from memory next morning. **Weekends:** merge A+B into one ~4-h capstone deep-work session if caught up; keep evenings for review; protect one full day off.

---

## 3. What changed (v12 → v12-DETAILED)
- **Every phase (0–9, incl. the 5A–5F track) elongated to v2 depth:** week-by-week, hour-budgeted resources with inline URLs, deliverables, whiteboard checkpoints, per-phase capstones, and interview self-tests — self-contained, no cross-references.
- **Threads (§5) expanded** to the same depth.
- **Resource index fused** into one topic-organized list (§14) — the earlier v10-core / v11-additions split is gone.
- All hours, calendar, ladder, COI, and feasibility from v12 preserved.

## 4. Program at a glance
**Block 1 — Foundations, imaging & molecular-ML core** (Sep 2026 → mid-Oct 2027): P0 → P1 → **P2 imaging (expanded)** → P2V → P3 → P4 → **Molecular-ML Track (5A–5F)**. Tier 0 → Tier 2.
**Phase Q — Quantum/QML** (parallel/after Tier-2): full depth; grounds 5D physics.
**Block 2 — Scientific agents, CV breadth & interview crescendo** (mid-Oct 2027 → mid-Feb 2028): P6 → P7 → P8. Tier 3.
**Threads (continuous from the Phase-1 self-test):** T (DSA+C++), M (implement-by-hand), R (research output). **Confusion Buffer** scaffolds every hard topic. **Application ladder** runs from Week 3.

---

## 5. Threads (continuous · ~550 hrs · start at the Phase-1 self-test, ~Week 10)
*Do T & M in Block B (first ~45 min a thread, then phase coding); R in the Evening. Flex to ~2 hrs combined on consolidation/heavy weeks. These exist so no interview surface ever feels shaky.*

### Thread T — DSA + C++ · ~5 hrs/week · ~275 hrs
**Stage T1 — C++ fundamentals (~40 hrs):** types, references/pointers, RAII, STL containers/algorithms, classes, templates, move semantics, smart pointers.
- learncpp.com (sections 1–17, hands-on) — **~24 hrs** — https://www.learncpp.com/
- The Cherno C++ (targeted: memory, pointers, move) — **~8 hrs** — https://www.youtube.com/@TheCherno
- cppreference (open while coding) — https://en.cppreference.com/ · Effective Modern C++ (Meyers) as reference · CppCon talks https://www.youtube.com/@CppCon — **~8 hrs**

**Stage T2 — DSA in C++, sustained (~235 hrs):** ~4–6 problems/week, spaced then timed.
- NeetCode 150 (pattern-first) — https://neetcode.io/ · LeetCode — https://leetcode.com/
- MIT 6.006 (algorithmic foundations) — https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ · Abdul Bari (algorithms) https://www.youtube.com/@abdul_bari
- Striver A2Z + SDE sheet — https://takeuforward.org/ · Sean Prashad patterns https://seanprashad.com/leetcode-patterns/ · DesignGurus (Grokking patterns) https://www.designgurus.io/
- CSES problem set + CP Handbook (for depth) — https://cses.fi/problemset/ · https://cses.fi/book/book.pdf
**Checkpoint:** core structures + standard patterns from memory in C++; a random LeetCode **medium in ~25–30 min, hard in ~40–45**.

### Thread M — Implement-by-hand ML · ~3 hrs/week · ~165 hrs · *(highest ROI + concept-cementer)*
From a **blank file, no autocomplete/AI**, one primitive in ~30–45 min, revisited spaced; add a weekly "build a small system" slot later. *(Validated: DeepMind's RE coding round asks you to implement a loss from a blank file.)*
- **Rotating primitives:** backprop through an MLP · scaled-dot-product + multi-head attention (+√dₖ) · Dice/Focal/Tversky · cross-entropy from logits · Layer/BatchNorm · stable log-softmax · a sampler (top-k/temperature) · a DDPM step · logistic/linear regression + gradients · k-means · PCA via SVD · SGD/Adam · positional encodings (sinusoidal/RoPE) · KV-cache · **a message-passing GNN layer** · **a small VQE circuit** · later **translate one to JAX**.
- References (read the mechanism, then close the tab): d2l.ai https://d2l.ai/ · labml.ai annotated https://nn.labml.ai/ · Annotated Transformer http://nlp.seas.harvard.edu/annotated-transformer/ · micrograd https://github.com/karpathy/micrograd · nanoGPT https://github.com/karpathy/nanoGPT · ML-From-Scratch https://github.com/eriklindernoren/ML-From-Scratch · minGPT https://github.com/karpathy/minGPT · tinygrad https://github.com/tinygrad/tinygrad · JAX https://jax.readthedocs.io/ · Equinox https://docs.kidger.site/equinox/
**Checkpoint:** implement any listed primitive cold, timed, and explain each line.

### Thread R — Research output, networking & applications · ~2 hrs/week · ~110 hrs
- **Reproduce-a-target-paper (monthly):** publicly reproduce one paper from a target team + write it up. — Papers with Code https://paperswithcode.com/ · How to Read a Paper (Keshav) https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf · Connected Papers https://www.connectedpapers.com/ · Papers We Love https://github.com/papers-we-love/papers-we-love
- **Write:** 1–2 short posts/month; a "why me for digital biology" narrative; two job-talk decks.
- **Publish:** aim a capstone at a venue — MIDL https://midl.io/ · MICCAI https://www.miccai.org/ · MLSB https://www.mlsb.io/ · LoG https://logconference.org/ · arXiv https://arxiv.org/ · bioRxiv https://www.biorxiv.org/ · OpenReview https://openreview.net/
- **Applications:** run the §8 ladder; from Tier 0 this thread carries your live loops.
**Checkpoint:** one reproduced paper + one write-up per month; a growing application pipeline.

---

# 6. THE PHASES (detailed)

## PHASE 0 — Fundamentals Rebuild: Stats, ML & DL · ~55 hrs · *Sep, Weeks 1–2* · → start applying (Tier 0)
**Goal:** rebuild crisp, out-loud command of the vocabulary and core ideas, intuition-first, building your living glossary as you go. By end you can explain ~15 fundamentals cold and you start applying to Tier-0 roles.

### Week 1 — Statistics + classic-ML start (~28 hrs)
- **StatQuest — Statistics playlist** (distributions, variance/SD, CLT, sampling, hypothesis testing & p-values, CIs, R², covariance/correlation, MLE, bootstrapping, power). Watch → write a glossary line + a tiny example for each. — **~14 hrs** — https://www.youtube.com/@statquest/playlists · https://statquest.org/
- **StatQuest — classic ML (start)** (bias–variance, cross-validation, confusion matrix, precision/recall, ROC/AUC & when PR is honest). — **~9 hrs**
- **Code drills** (Block B): simulate the CLT; a permutation test + bootstrap CI; a bias-variance curve. — **~5 hrs**

**Deliverable:** the first ~40 glossary entries (your terminology war-chest v1).
**Whiteboard checkpoint:** define a p-value and a 95% CI precisely; state the CLT; derive the MLE mean of a Gaussian; draw the bias-variance trade-off.

### Week 2 — Classic ML + DL fundamentals + self-test (~27 hrs)
- **StatQuest — classic ML (finish):** linear/logistic regression; ridge vs lasso & **why L1→sparsity**; trees/RF/AdaBoost/GBM/XGBoost; SVMs & kernels; k-means/hierarchical; PCA; naive Bayes; entropy & mutual information. — **~11 hrs**
- **DL fundamentals** — StatQuest NN + 3B1B "Neural Networks" ch 1–2 (neurons, layers, gradient descent, the shape of learning). — **~9 hrs** — 3B1B NN https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **Code drills:** logistic regression + L1/L2 from scratch (watch weights zero out); ROC & PR curves on an imbalanced toy set; a 1-neuron gradient-descent loop. — **~7 hrs**

**Deliverable (0.4):** an articulation cheat-sheet (each concept in 3 plain sentences) + glossary v1. Reference: StatQuest *Illustrated Guide to ML*.
**Phase 0 self-test (Feynman gate):** explain, out loud and with no notes — p-value, CI, CLT, MLE, bias-variance, ROC/AUC vs PR, why L1→sparsity, GBM vs RF, entropy/cross-entropy, gradient descent, over/underfitting. **→ START APPLYING (Tier 0):** refresh CV/LinkedIn, first 3–5 quality applications.

---

## PHASE 1 — Mathematics + Neural Networks from First Principles · ~210 hrs · *Sep–Oct, Weeks 3–12* · a genuine build — do NOT rush
**Goal:** real mathematical maturity — the thing that breaks the vanilla-model ceiling and lets you explain *why* a model behaves as it does. Visual → rigorous → derive by hand, spaced. By end you can hand-derive backprop, implement micrograd from memory, and whiteboard attention with the √dₖ rationale — the gate that switches the Threads on.

### Weeks 3–4 — Linear algebra (~72 hrs across the phase; front-loaded here)
- **3B1B — Essence of Linear Algebra** (build geometric intuition first; watch twice if needed) — **~8 hrs** — https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- **MIT 18.06 (Strang)** — selected lectures L1,3,5,6,9,10,14,15,16,21,22,25,29,30 + problems — **~30 hrs** — https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/
- **MIT 18.065** — SVD / PCA / low-rank / gradient-descent-through-a-LA-lens sections — **~18 hrs** — https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/
- Reference kept open: Matrix Cookbook https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf · (optional) Imperial Math-for-ML LA https://www.coursera.org/learn/linear-algebra-machine-learning
- **Code drills (Block B):** vectors/matmul/transformations from scratch; eigendecomposition & SVD; truncated-SVD image compression; least-squares projection. — **~16 hrs**

**Deliverable:** an LA notebook (SVD-PCA + low-rank compression + a projection) with a written "geometry of each operation" note.
**Whiteboard checkpoint:** derive SVD from the eigendecomposition of AᵀA; show why AᵀA is PSD; explain geometrically what a rank-r approximation does (Eckart–Young); derive the least-squares normal equations.

### Week 5 — 🛑 CONSOLIDATION (~18 hrs, no new material)
Re-derive from memory (matmul-as-composition, determinant meaning, SVD-from-eigendecomposition, projection); re-implement SVD-PCA from a blank file; full Anki/glossary catch-up. **Do not skip.**

### Weeks 6–7 — Calculus + matrix calculus (~26 hrs)
- **3B1B — Essence of Calculus** (derivatives, chain rule, the fundamental ideas) — **~6 hrs** — https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
- **Matrix Calculus for Deep Learning (Parr & Howard)** — derive gradients of vector/matrix expressions — **~12 hrs** — https://explained.ai/matrix-calculus/ · (optional) Imperial Multivariate Calc https://www.coursera.org/learn/multivariate-calculus-machine-learning
- **Code drills:** gradient of a linear layer (∂L/∂W=δxᵀ) + numerical gradient check; softmax+cross-entropy forward/backward. — **~8 hrs**

**Deliverable:** a from-scratch, gradient-checked linear layer + softmax-CE layer.
**Whiteboard checkpoint:** derive backprop through a linear layer with matrix calculus; derive ∂(softmax+CE)/∂logits = p − y; Jacobian vs Hessian.

### Weeks 8–9 — Probability & statistics (~50 hrs)
- **Harvard Stat 110 (Blitzstein)** — targeted: probability, conditional probability & Bayes, random variables, expectation/variance, key distributions, conditional expectation — **~28 hrs** — https://projects.iq.harvard.edu/stat110/youtube · book (free) https://probabilitybook.net/
- **MIT RES.6-012 (Intro to Probability)** — as a second teacher where Stat 110 is thin — **~12 hrs** — https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/
- **Code drills:** simulate distributions; MLE fitting; a small Bayesian update; conditional-expectation-as-projection demo. — **~10 hrs**

**Deliverable:** a probability notebook (Bayes example + MLE fit + CLT simulation).
**Whiteboard checkpoint:** Bayes' theorem with each term named; E[X|Y] as an orthogonal projection; MLE for a Gaussian; why NLL minimization = MLE and cross-entropy = KL up to a constant.

### Weeks 10–11 — Neural networks from first principles (Karpathy Zero-to-Hero) (~62 hrs) · PROTECT
- **Karpathy "Neural Networks: Zero to Hero"** — all 10 videos, **re-implement each without looking at his code** (micrograd → makemore incl. BatchNorm internals + manual backprop + WaveNet → GPT from scratch → tokenizer). — **~54 hrs** — https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ · hub https://karpathy.ai/zero-to-hero.html · repo https://github.com/karpathy/nn-zero-to-hero
- **Thread M kick-off (Week 10):** reimplement backprop-through-MLP and multi-head attention from a blank file. — **~8 hrs**

**Deliverable:** micrograd + a small GPT, both built from scratch, in your `learning-log` repo.
**Phase 1 self-test (gate → Threads T/M/R begin):** hand-derive backprop for a 2-layer MLP; implement micrograd from memory; explain why BatchNorm helps and what breaks without it; whiteboard scaled-dot-product attention and derive the √dₖ scaling.

### Week 12 — 🛑 CONSOLIDATION (~18 hrs + light threads)
Re-derive backprop, attention+√dₖ, softmax-CE gradient, SVD from a blank page; re-implement micrograd-MLP and multi-head attention; Anki/glossary cleanup; a teach-a-junior note on attention. Threads continue light.

---

## PHASE 2 — Applied CV & Biomedical Image Segmentation (EXPANDED) · ~195 hrs core (+ Capstone 1 ~25) + expansion ~165 hrs · *Dec 2026 → mid-Mar 2027* · → Tier 0/1
**Goal:** own MRI / fluorescence / whole-slide segmentation end-to-end — physics → preprocessing → architectures → losses → 3D → uncertainty → evaluation — then extend into the discovery-imaging methods (phenomics, atlas registration, WSI engineering, pathology foundation models). By end you can whiteboard nnU-Net's design choices cold, defend each, and stand up a phenomics/WSI pipeline on public data.

### Week 1 — Imaging physics (~30 hrs) — *most engineers skip this and it shows in their model design*
- **Allen Elster — Questions & Answers in MRI** (browse; the reference you'll return to) — **~8 hrs** — https://mriquestions.com/index.html
- **OHBM Educational — MRI physics** playlist — **~8 hrs** — https://www.youtube.com/@ohbmonline/playlists
- **Paul Callaghan — "Magnetic Resonance" (Otago)**, first 4 lectures — **~6 hrs** — https://www.youtube.com/playlist?list=PLqEMVgX2VgZcG_xR1QRdsuc-X_zEgMx-Z
- **iBiology — fluorescence microscopy primer** (your tissue work needs this) — **~8 hrs** — https://www.youtube.com/playlist?list=PLF513KEDjY9YBNhwMfX6jMl3PlpdW9TMP

**Deliverable:** 1-page note — T1 vs T2 vs FLAIR vs T1CE (tissue appearance, why each exists, artifacts to preprocess: motion, bias field, partial volume); for fluorescence: photobleaching, channel bleed-through, autofluorescence and their effect on segmentation.
**Whiteboard checkpoint:** which artifacts must be corrected before training and why; how MRI contrast maps to what a network can/can't learn.

### Week 2 — CNNs, U-Net family, convolution math (~35 hrs)
- **Stanford CS231n** (focus L5, 9, 11 + notes) — **~14 hrs** — https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv · notes https://cs231n.github.io/
- **Convolution arithmetic (Dumoulin & Visin)** — canonical for shapes/strides/padding/dilation, 2D & 3D — **~5 hrs** — https://arxiv.org/abs/1603.07285 · animations https://github.com/vdumoulin/conv_arithmetic
- **U-Net family — read in order, derive output sizes by hand** — **~16 hrs** — U-Net https://arxiv.org/abs/1505.04597 · V-Net (3D) https://arxiv.org/abs/1606.04797 · Attention U-Net https://arxiv.org/abs/1804.03999 · nnU-Net (read twice + supplementary) https://www.nature.com/articles/s41592-020-01008-z

**Deliverable:** a hand-computed layer-by-layer shape table for a U-Net on 256×256 input.
**Whiteboard checkpoint:** draw U-Net from memory; compute output size at every layer; explain why skip connections work; explain why nnU-Net's "no new architecture" thesis is profound.

### Week 3 — Loss functions for medical segmentation (your 95/5 imbalance) (~30 hrs) — *where segmentation pipelines die; own it cold*
- **Survey of segmentation losses (Jadon)** — derive each — **~6 hrs** — https://arxiv.org/abs/2006.14822
- Generalized Dice (Sudre) https://arxiv.org/abs/1707.03237 · Focal Tversky https://arxiv.org/abs/1810.07842 · Boundary Loss https://arxiv.org/abs/1812.07032 — **~9 hrs**
- **From-scratch PyTorch (Thread M):** Dice, GDL, Focal, Tversky, Focal Tversky, Boundary, compound DiceCE; unit-test on synthetic imbalanced data; **plot gradients vs class ratio**; document when each helps. — **~15 hrs**

**Deliverable:** a tested losses library + a "gradient vs class-ratio" plot with a when-to-use guide.
**Whiteboard checkpoint:** derive the Dice gradient and explain its instability on empty masks; why Focal/Tversky help severe imbalance.

### Week 4 — nnU-Net deep dive + MONAI + microscopy (~45 hrs) — *the biggest hands-on block; nnU-Net stops being a black box*
- **MONAI tutorials** (clone, run, modify the brain-seg notebooks) — **~15 hrs** — https://github.com/Project-MONAI/tutorials · videos https://www.youtube.com/@projectmonai/videos
- **nnU-Net v2 — read the source** (teaches more than the paper) — **~8 hrs** — https://github.com/MIC-DKFZ/nnUNet · Kitware comparison https://www.kitware.com/developing-custom-3d-medical-image-segmentation-solutions-using-out-of-the-box-pipelines-in-monai/
- **DigitalSreeni (Python for Microscopists)** — the most microscopy-native channel; ~6–8 code-along videos mapped to your bottlenecks — **~14 hrs** — https://www.youtube.com/@DigitalSreeni/playlists · code https://github.com/bnsreenu/python_for_microscopists
- **AI for Medical Diagnosis (Rajpurkar)** — audit; the brain-MRI U-Net + evaluation modules — **~8 hrs** — https://www.coursera.org/learn/ai-for-medical-diagnosis

**Deliverable:** reproduce nnU-Net on a public brain dataset (BraTS subset / Decathlon task); write up which defaults it picked, why, and what you'd change for fluorescence microscopy.
**Whiteboard checkpoint:** walk nnU-Net's fingerprinting → configuration; when it beats fancier architectures and when it doesn't.

### Week 5 — Transformer-based segmentation + foundation models (~20 hrs)
- **ViT (re-derive attention math)** https://arxiv.org/abs/2010.11929 — **~4 hrs**
- **TransUNet / Swin-UNet / UNETR / Swin UNETR** — **~10 hrs** — https://arxiv.org/abs/2102.04306 · https://arxiv.org/abs/2105.05537 · https://arxiv.org/abs/2103.10504 · https://arxiv.org/abs/2201.01266
- **SAM / MedSAM** https://arxiv.org/abs/2304.02643 · https://www.nature.com/articles/s41467-024-44824-z — **~6 hrs**
- **MedNeXt** (a ConvNet that beats transformers on many medical tasks — keeps you honest) https://arxiv.org/abs/2303.09975 — read

**Whiteboard checkpoint:** inductive biases CNN vs ViT vs hybrid — when each wins; foundation models in low-data medical regimes; SAM's promptable design and what MedSAM changes.

### Week 6 — 3D, uncertainty, evaluation + Capstone 1 (~35 hrs + Capstone 25)
- **TorchIO** (augmentation done right for volumes) — **~10 hrs** — https://torchio.readthedocs.io/
- **Uncertainty (Yarin Gal — MC Dropout, deep ensembles, evidential)** — **~8 hrs** — https://www.youtube.com/results?search_query=yarin+gal+uncertainty+deep+learning
- **Metrics Reloaded** (the definitive metric-choice paper) — **~5 hrs** — https://www.nature.com/articles/s41592-023-02151-z · Domain shift (Stanford AIMI) https://aimi.stanford.edu/education/educational-resources
- **3D-conv whiteboard drill:** input (D=64,H=128,W=128,C=4), kernel (3,3,3), stride (1,2,2), padding (1,1,1), 32 out — compute output shape + param count by hand; repeat for transposed & dilated. — **~3 hrs**

**CAPSTONE 1 (~25 hrs) · public data · Tier-0/1 gate:** self-contained notebook + 3-page write-up on a public brain set (BraTS subset / Decathlon): (1) preprocessing with physics justification, (2) architecture ablation (3), (3) loss ablation, (4) uncertainty + failure-mode analysis; implement losses + one block from scratch. Datasets: Medical Decathlon http://medicaldecathlon.com/ · BraTS https://www.synapse.org/brats · TCIA https://www.cancerimagingarchive.net/ · IXI https://brain-development.org/ixi-dataset/ · LIVECell https://github.com/sartorius-research/LIVECell · BBBC https://bbbc.broadinstitute.org/
**Capstone-1 self-test:** whiteboard U-Net forward + skips with shapes; derive Dice gradient + empty-mask instability; why nnU-Net usually wins; for a new dataset, diagnose bias-field / imbalance / label-noise / domain-shift.

### Imaging expansion modules (~165 hrs) — the discovery-imaging skill set (public data throughout)

**2.6 — Cell/nucleus segmentation + phenomics (~35 hrs).**
- Cellpose https://www.cellpose.org/ · StarDist https://github.com/stardist/stardist — **~12 hrs** (run both on a public cell set).
- Cell Painting morphological profiling with **CellProfiler**; dose–response/EC50 + ROC-AUC validation — **~13 hrs** — https://cellprofiler.org/
- Datasets: BBBC https://bbbc.broadinstitute.org/ · **JUMP-CP** https://jump-cellpainting.broadinstitute.org/ · **Recursion RxRx** (public phenomics — direct Recursion-target match) https://www.rxrx.ai/ — **~10 hrs** (build a small profiling pipeline).
*Deliverable:* segment → extract morphological features → fit a dose–response/AUC readout on a public set. *Checkpoint:* what a Cell-Painting profile is; why EC50/AUC is the readout; batch effects in phenomics.

**2.7 — 3D segmentation + atlas registration (~30 hrs).**
- 3D sub-region segmentation + registration to a reference atlas — **Allen Brain Atlas** https://atlas.brain-map.org/ · **BrainGlobe/brainreg** https://brainglobe.info/ · ANTsPy https://github.com/ANTsX/ANTsPy · SimpleITK https://simpleitk.org/ — **~30 hrs**.
*Deliverable:* register a public mouse-brain volume to an atlas and quantify a per-region readout. *Checkpoint:* rigid vs affine vs deformable registration; what an atlas buys you; how registration error propagates to region metrics.

**2.8 — Whole-slide / large-image engineering (~25 hrs).**
- Tiling, pyramidal I/O, zarr/dask for 20–50 GB images; QC with QuPath/napari — OpenSlide https://openslide.org/ · **TIAToolbox** https://github.com/TissueImageAnalytics/tiatoolbox · CLAM https://github.com/mahmoodlab/CLAM · QuPath https://qupath.github.io/ · napari https://napari.org/ — **~25 hrs**.
*Deliverable:* a tiling + tiled-inference pipeline on a public WSI with stitched output. *Checkpoint:* why naive full-image loading fails; how to keep tile context; memory/throughput trade-offs.

**2.9 — Stain normalization + MIL/weak-supervision + active learning (~25 hrs).**
- Macenko/Reinhard/Vahadane normalization — torchstain https://github.com/EIDOSLAB/torchstain — **~8 hrs**.
- Multiple-instance learning for weak WSI labels — CLAM https://github.com/mahmoodlab/CLAM — **~10 hrs**.
- Active learning to cut annotation — modAL https://github.com/modAL-python/modAL — **~7 hrs**.
*Deliverable:* an MIL slide-level classifier on a public WSI set with stain-norm ablation. *Checkpoint:* why stain variation breaks models; the MIL assumption; how active learning chooses samples.

**2.10 — Pathology foundation models & SSL (~40 hrs) · strategic.**
- DINOv2/MAE self-supervision — DINOv2 https://github.com/facebookresearch/dinov2 · MAE https://arxiv.org/abs/2111.06377 — **~14 hrs**.
- Use a pathology foundation model as feature extractor + fine-tune — **UNI** https://github.com/mahmoodlab/UNI · CONCH · Virchow (Nature Medicine 2024) https://huggingface.co/paige-ai — **~26 hrs** (read the Virchow paper; run UNI embeddings + a linear/fine-tune head on a public task).
*Deliverable:* compare supervised CNN features vs foundation-model embeddings on a public phenotypic/pathology task.
*Checkpoint:* what SSL objective Virchow/DINOv2 use; why FM embeddings win in low-data; when fine-tuning beats linear-probe.

**IMAGING-TRACK CAPSTONE (public):** tile a large public WSI/microscopy image → segment → extract morphological/phenotypic features → fit a dose–response/AUC readout; compare CNN features vs foundation-model embeddings. One artifact showcasing segmentation + phenomics + foundation-model transfer — a MIDL/MICCAI-workshop candidate (Thread R).

**PHASE 2 interview self-test:** the Capstone-1 set **plus** — what a Cell-Painting profile encodes and its batch effects; registration types + error propagation; why WSI needs tiling/zarr and how to preserve context; the MIL assumption + stain-norm rationale; the SSL objective behind a pathology foundation model and when its embeddings beat a supervised CNN. **→ Tier 0/1 (imaging + phenomics → Recursion/PathAI credible).**

---

## PHASE 2V — Imaging & Video Extension (life-sciences-native) · ~70 hrs (+ Capstone 2 ~30) · *mid-Mar – early Apr 2027* · → Tier 1
**Goal:** extend imaging into video/temporal — unlocking phenomics/cell-imaging over time and behavioral analysis. By end you can build a pose→behavior or cell-track→lineage pipeline and reason about tracking-by-detection vs joint methods.

### Weeks 1–2 — Video & temporal models (~40 hrs)
- **Video transformers & motion** — VideoMAE https://arxiv.org/abs/2203.12602 · TimeSformer https://arxiv.org/abs/2102.05095 · ViViT https://arxiv.org/abs/2103.15691 · SlowFast https://arxiv.org/abs/1812.03982 — **~14 hrs**
- **Segmentation/tracking over time** — **SAM 2** (memory mechanism) https://arxiv.org/abs/2408.00714 · repo https://github.com/facebookresearch/sam2 · RAFT (optical flow) https://arxiv.org/abs/2003.12039 — **~12 hrs**
- **Multi-object tracking** — ByteTrack https://arxiv.org/abs/2110.06864 · OC-SORT https://arxiv.org/abs/2203.14360 · BoT-SORT https://arxiv.org/abs/2206.14651 · HOTA metric https://arxiv.org/abs/2009.07736 — **~14 hrs** — benchmarks: MOTChallenge https://motchallenge.net/ · DAVIS https://davischallenge.org/ · YouTube-VOS https://youtube-vos.org/

**Whiteboard checkpoint:** SAM 2's memory mechanism; tracking-by-detection vs joint detection-tracking; when a video transformer beats frame-wise + tracking; the HOTA metric.

### Weeks 3 — Life-sciences-native video (~30 hrs) — *the differentiator*
- **Pose estimation** — DeepLabCut https://www.deeplabcut.org/ (paper https://www.nature.com/articles/s41593-018-0209-y) · SLEAP https://sleap.ai/ — **~14 hrs**
- **Cell tracking & lineage** — Cellpose https://www.cellpose.org/ → TrackMate https://imagej.net/plugins/trackmate/ / Ultrack https://github.com/royerlab/ultrack — **~16 hrs** — Cell Tracking Challenge http://celltrackingchallenge.net/

**CAPSTONE 2 (video) · public data · Tier 1:** DeepLabCut/SLEAP pose → temporal model → behavior classification; **or** Cellpose seg → TrackMate/Ultrack tracking + lineage; SAM 2 for mask propagation. Datasets: CalMS21 https://data.caltech.edu/records/s0vdx-0k302 · MABe https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022 · Cell Tracking Challenge http://celltrackingchallenge.net/
**Phase 2V self-test:** SAM 2 memory; tracking-by-detection vs joint; when a video transformer wins; cell-lineage failure modes. **→ Tier 1 unlocked (phenomics/behavioral imaging).**

---

## PHASE 3 — Modern Deep Learning Theory + Foundation Architectures · ~165 hrs · *April – mid-May 2027* · → Tier 1 (broad)
**Goal:** stop treating transformers and diffusion as black boxes. Derive every layer of attention, understand optimization dynamics and modern training tricks, and defend architecture choices on theoretical grounds. This deepens skills you already use *and* is the prerequisite for Phase 4 and the generative parts of the molecular track (5C).

### Week 1 — Optimization dynamics (~30 hrs)
- **Ruder — gradient-descent overview** (SGD → momentum → Adam family) — **~6 hrs** — https://www.ruder.io/optimizing-gradient-descent/
- **Boyd EE364A — Convex Optimization** (selected: convex sets/functions, gradient/Newton, duality intuition) — **~20 hrs** — https://www.youtube.com/playlist?list=PL3940DD955CDF0622 · book https://web.stanford.edu/~boyd/cvxbook/
- **Code drills:** implement SGD/Momentum/Adam from scratch; visualize on a 2D loss surface; a LR-warmup/schedule experiment. — **~4 hrs**

**Whiteboard checkpoint:** why momentum accelerates; what Adam's second moment does; what warmup/schedules change in training dynamics; convex vs non-convex and why DL still works.

### Week 2 — Information theory (~20 hrs)
- **MacKay — Information Theory, Inference & Learning (ch 1–6)** — entropy, KL, mutual information, coding — **~20 hrs** — https://www.inference.org.uk/itila/book.html

**Whiteboard checkpoint:** derive cross-entropy from KL; NLL minimization = MLE; where mutual information appears in contrastive losses (InfoNCE).

### Week 3 — Transformers in depth + modern variants (~44 hrs)
- **Stanford CS25 — Transformers United** (6–8 talks) — **~16 hrs** — https://web.stanford.edu/class/cs25/ · playlist https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM
- **The Annotated Transformer** (code + math line by line) + **Lilian Weng** blog — **~16 hrs** — http://nlp.seas.harvard.edu/annotated-transformer/ · https://lilianweng.github.io/
- **Modern tricks (~30 min each):** RoPE https://arxiv.org/abs/2104.09864 · RMSNorm https://arxiv.org/abs/1910.07467 · FlashAttention https://arxiv.org/abs/2205.14135 · GQA/MQA https://arxiv.org/abs/2305.13245 · SwiGLU https://arxiv.org/abs/2002.05202 · MoE/Switch https://arxiv.org/abs/2101.03961 — **~12 hrs**

**Deliverable:** a from-scratch transformer block (multi-head attention as one matmul, RoPE, RMSNorm, SwiGLU) with a param-count derivation.
**Whiteboard checkpoint:** derive scaled dot-product attention; multi-head as a single matmul; RoPE geometrically; why pre-LN beats post-LN; compute a block's param count from d_model, n_heads, FFN ratio.

### Week 4 — Vision transformers + modern ConvNets (~27 hrs)
- **ViT lectures (Lucas Beyer)** — **~8 hrs** — https://www.youtube.com/results?search_query=lucas+beyer+vision+transformer
- **ConvNeXt** https://arxiv.org/abs/2201.03545 · **DINOv2** (SSL, label-scarce) https://arxiv.org/abs/2304.07193 · **MAE** https://arxiv.org/abs/2111.06377 — **~12 hrs**
- **HF Computer Vision course** (units new to you) — **~7 hrs** — https://huggingface.co/learn/computer-vision-course

**Whiteboard checkpoint:** ViT patch embedding; CNN vs ViT inductive biases; why SSL helps in low-data (ties to Phase 2.10).

### Week 5 — Diffusion + generative modeling theory (~44 hrs) · PROTECT — feeds 5C
- **Stanford CS236 — Deep Generative Models** (L1–3, 6–8, 11–13) — **~22 hrs** — https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8
- **DDPM** https://arxiv.org/abs/2006.11239 · **Yang Song — score-based unification + blog** https://yang-song.net/blog/2021/score/ · **Lilian Weng diffusion** https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ — **~14 hrs**
- **HF Diffusion course** (units 1–2, hands-on) — **~8 hrs** — https://huggingface.co/learn/diffusion-course

**Deliverable:** a minimal DDPM trained on a toy set (implement the forward/reverse + the ε-loss from scratch — Thread M).
**Phase 3 self-test:** implement attention on a whiteboard in 5 min; derive the diffusion loss (variational + score-matching views); why LayerNorm not BatchNorm in transformers; positional encodings compared (sinusoidal/learned/RoPE/ALiBi); compute a transformer block's param count.

---

## PHASE 4 — Full-Stack / Production AI Systems + MLOps · ~165 hrs · *mid-May – late Jun 2027*
**Goal:** upgrade from "I built it" to "I can defend every architectural choice and design the next-gen version" across LLM systems, RAG, MLOps, distributed training, and serving — and be fluent in ML system design (a core interview surface, and directly useful to your critic-agent project).

### Week 1 — LLM internals (~34 hrs)
- **Stanford CS336 — Language Modeling from Scratch** (currently the best LLM-systems course) — **~26 hrs** — https://stanford-cs336.github.io/spring2024/ · playlist https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_
- **HF NLP course** (units new to you) — **~8 hrs** — https://huggingface.co/learn/nlp-course

**Whiteboard checkpoint:** tokenization → embedding → blocks → head; KV-cache and why it matters; sampling (greedy/top-k/nucleus/temperature).

### Week 2 — RAG at senior level (~28 hrs)
- **Hamel Husain — "What we learned from a year of building with LLMs" (I–II)** — **~8 hrs** — https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/
- **Eugene Yan — LLM patterns** + **Anthropic — contextual retrieval** — **~8 hrs** — https://eugeneyan.com/writing/llm-patterns/ · https://www.anthropic.com/news/contextual-retrieval
- **Reranking, hybrid search, query rewriting, evals** — LlamaIndex https://docs.llamaindex.ai/ · RAGAS https://docs.ragas.io/ · DSPy https://dspy.ai/ — **~12 hrs**

**Deliverable:** a RAG system with hybrid retrieval + reranking + a RAGAS eval harness on public docs.
**Whiteboard checkpoint:** chunking/embedding/retrieval/rerank trade-offs; how you'd evaluate a RAG system honestly; where a critic/verifier agent improves it (ties to your project).

### Week 3 — MLOps + ML systems design (~44 hrs) · KEEP
- **Chip Huyen — Designing ML Systems / interviews book** — **~18 hrs** — https://huyenchip.com/ml-interviews-book/contents/8.1.1-ml-systems-design.html
- **Stanford CS329S — ML Systems Design** — **~14 hrs** — https://stanford-cs329s.github.io/
- **Made With ML** (hands-on) + **Full Stack Deep Learning** — **~12 hrs** — https://madewithml.com/ · https://fullstackdeeplearning.com/course/2022/

**Deliverable:** a written ML-system-design doc for a real scenario (e.g., a molecular-property serving system, or a brain-seg pipeline ingesting new-institution data weekly with domain-shift handling).
**Whiteboard checkpoint:** design a training→serving→monitoring loop with data/feature/model/serving layers; drift detection and retraining triggers.

### Week 4 — Distributed training + inference optimization (~26 hrs)
- **Sebastian Raschka** (distributed training + efficient fine-tuning) — **~8 hrs** — https://magazine.sebastianraschka.com/
- **HF — efficient training on multiple GPUs** — **~6 hrs** — https://huggingface.co/docs/transformers/en/perf_train_gpu_many
- **vLLM (paper + run it)** + quantization (GPTQ/AWQ/GGUF/FP7) + **ZeRO** — **~12 hrs** — https://arxiv.org/abs/2309.06180 · ZeRO https://arxiv.org/abs/1910.02054

**Whiteboard checkpoint:** data vs tensor vs pipeline parallelism; what ZeRO shards; how KV-cache + paged attention speed inference; the accuracy/latency of quantization.

### Week 5 — Eval / observability / safety + systems-design prep (~33 hrs)
- **Anthropic — Constitutional AI** + **Eugene Yan — LLM evals** — **~10 hrs** — https://www.anthropic.com/news/constitutional-ai · https://eugeneyan.com/writing/llm-evaluators/
- **Observability** — run one of LangSmith / Phoenix / Helicone on a real RAG system — **~8 hrs**
- **Systems-design interview prep** — whiteboard, recorded, self-critiqued: "RAG assistant for HCPs, 10K queries/day on 50K docs"; "brain-seg pipeline with weekly new-institution data + domain-shift handling"; "retraining + monitoring for a drug-discovery digital twin." — **~15 hrs** — Chip Huyen book https://huyenchip.com/ml-interviews-book/

**Phase 4 self-test:** design an LLM-eval harness (incl. an LLM-as-judge with its pitfalls); a full ML-system-design whiteboard in 45 min; defend a serving/monitoring architecture; where a critic-agent adds value and how you'd evaluate *it*.

---

## PHASE 5 · MOLECULAR-ML / DRUG-DISCOVERY TRACK (Phases 5A–5F)

**Placement:** Weeks 43–58 of v12 (~late Jun → mid-Oct 2027, **15 weeks**) · **~450 hrs** · **PRIORITY: high** — this is the differentiator that turns "strong imaging engineer" into "ML researcher for drug discovery," and the unlock for **Tier 2** (D. E. Shaw Research, Isomorphic Labs, DeepMind-science, Recursion/Insitro).

**Goal of the whole track.** By the end you can, on **public data**, cold and unaided: build and evaluate molecular property predictors across the full representation ladder (fingerprint → D-MPNN → 3D-equivariant); pretrain and finetune a molecular foundation model; generate diverse high-reward molecules with GFlowNets and equivariant diffusion; run and reason about the physics stack (conformer → docking → FEP → MD) and place every method on the accuracy/cost ladder; put **valid, calibrated uncertainty** on any prediction (conformal, GP/SVGP, ensembles); and ship the applied discovery methods a small-molecule squad actually uses (retrosynthesis, DEL, multitask, federated). Every claim here must be defensible in a DESRES-grade panel — the deliverables and whiteboard checkpoints exist to force that.

**Prerequisites & sequencing.** Needs Phase 1 (linear algebra, calculus/matrix-calculus, probability, backprop) and benefits from Phase 3 (GNN/attention/diffusion theory sits under 5A and 5C). **Runs alongside Phase Q** — the quantum-chemistry work grounds 5D's physics (why DFT/FEP are accurate and where they break). Do the sub-phases in order: **5A is foundational** (everything else assumes molecular representations); 5D is the heaviest and most DESRES-relevant; 5E is cross-cutting and can be pulled earlier if a UQ-heavy role appears.

**How the hours land in your day (recap of the v12 rhythm).** Block A 06:00–08:00 = the hard new theory/derivation of the day; Block B 08:30–10:30 = build it in code + the day's Thread-M/Thread-T work; Evening 20:30–22:30 = paper reading, re-derivation, Anki, planning. Weekends carry the capstone deep-work (uninterrupted blocks) and absorb spillover. At ~30 structured hrs/week, ~450 hrs ≈ **~15 weeks**.

**Sub-phase summary**

| Sub-phase | Focus | Weeks (of track) | Approx. hrs |
|---|---|---|---|
| **5A** | Geometric & molecular ML core: GNNs → D-MPNN → 3D-equivariant | 1–3 | ~90 |
| **5B** | Molecular foundation models & self-supervised pretraining | 4–5 | ~50 |
| **5C** | Generative chemistry: GFlowNets + equivariant diffusion | 6–8 | ~70 |
| **5D** | Physics-based structure & simulation: conformer → docking → FEP → MD | 9–12 | ~110 |
| **5E** | Uncertainty quantification & probabilistic ML | 13–14 | ~55 |
| **5F** | Applied cheminformatics: retrosynthesis, DEL, multitask, federated, VS | 15–16 | ~75 |

---

## 5A — Geometric & Molecular ML Core · ~90 hrs · PRIORITY: foundational
*Everything downstream assumes you can turn a molecule into a good representation and reason about it. This is where that happens.*

**Goal:** own molecular representation learning from fingerprints to message-passing to 3D-equivariant nets, and know **cold when each wins**. By end you can implement a message-passing layer from a blank file, train Chemprop and an equivariant GNN that beat a fingerprint baseline on a public ADMET benchmark, and defend every representation choice.

### Week 1 — Graph ML foundations + molecular representations (~30 hrs)
Build the graph-learning spine and set the baseline every later model must beat.
- **Stanford CS224W (Leskovec)** — selected lectures: intro & node embeddings, the GNN lectures, message passing, and GNN expressiveness (Weisfeiler-Lehman). Watch → re-derive the message-passing update; do the associated Colabs. — **~14 hrs** — https://web.stanford.edu/class/cs224w/ · playlist https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn
- **PyG "Introduction by Example" + official Colabs** — build GCN / GraphSAGE / GAT on a toy graph task; understand batching of graphs. — **~6 hrs** — https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html
- **Molecular representations primer** — fingerprints (ECFP/Morgan) vs physicochemical descriptors vs learned embeddings; RDKit getting-started + a fingerprint/descriptor tutorial. — **~5 hrs** — RDKit https://www.rdkit.org/docs/GettingStartedInPython.html · DeepChem tutorials https://deepchem.io/tutorials/
- **Baseline build** — featurize a public ADMET task with Morgan fingerprints + RDKit descriptors; train XGBoost/RF with **scaffold splits** (not random — this is the honest split for molecules). — **~5 hrs** — Therapeutics Data Commons (ADMET benchmarks) https://tdcommons.ai/ · MoleculeNet https://moleculenet.org/

**Deliverable:** a *baselines* notebook (fingerprint + descriptor models) on one TDC ADMET task with scaffold splits and proper metrics (RMSE/R² or AUROC) — this is the yardstick every model in 5A–5F must beat.
**Whiteboard checkpoint:** write the message-passing update (aggregate → update) and prove permutation-invariance of the readout; state the Weisfeiler-Lehman bound on GNN expressiveness and what it means practically; contrast fingerprints vs learned embeddings (when does a fingerprint+GBM still win?).

### Week 2 — Message-passing for molecules (D-MPNN / Chemprop) + graph transformers (~30 hrs)
The workhorse of production molecular property prediction.
- **Chemprop / D-MPNN** — read "Analyzing Learned Molecular Representations for Property Prediction" (Yang et al. 2019), then install and train Chemprop on your Week-1 task; ablate directed vs undirected messages and with/without RDKit-feature concatenation. — **~12 hrs** — https://github.com/chemprop/chemprop
- **Implement a message-passing GNN layer from scratch** (pure PyTorch or PyG `MessagePassing`) on a small molecular set — forward + backward, gradient-checked. This is a Thread-M anchor. — **~10 hrs**
- **Graph transformers** — read one of Graphormer / GraphGPS / TokenGT; understand how graph structure (centrality, spatial, edge encodings) is injected into attention and when a graph transformer beats an MPNN. — **~8 hrs** — Graphormer https://arxiv.org/abs/2106.05234 · GraphGPS https://arxiv.org/abs/2205.12454

**Deliverable:** Chemprop **and** your from-scratch MPNN beating the Week-1 baseline on the TDC task; a one-page note on directed-MPNN design (why bond/edge-centered messages reduce "tottering") and when you'd reach for a graph transformer.
**Whiteboard checkpoint:** derive the D-MPNN edge-message update and explain the tottering argument; explain how a graph transformer encodes structure into attention; state the over-smoothing problem in deep GNNs and two fixes.

### Week 3 — 3D / equivariant molecular networks (~30 hrs)
Where geometry enters — and where the DESRES/quantum-chemistry world lives.
- **Geometric Deep Learning (Bronstein et al.)** — the symmetry/equivariance lectures; the "5 Gs" framing. Watch → be able to state invariance vs equivariance precisely. — **~8 hrs** — https://geometricdeeplearning.com/lectures/ · proto-book https://arxiv.org/abs/2104.13478
- **Equivariant nets — read in order, derive the equivariance property each time:** SchNet (continuous-filter conv) → DimeNet++ (directional/angular) → EGNN (simple E(n)-equivariance) → NequIP/MACE (tensor-product message passing for potentials). — **~14 hrs** — SchNet https://arxiv.org/abs/1706.08566 · DimeNet++ https://arxiv.org/abs/2011.14115 · EGNN https://arxiv.org/abs/2102.09844 · NequIP https://www.nature.com/articles/s41467-022-29939-5 · MACE https://arxiv.org/abs/2206.07697
- **e3nn tutorial + run an equivariant model on QM9** — train SchNet/EGNN-style on a QM9 property; verify equivariance empirically (rotate input → output transforms correctly). — **~8 hrs** — e3nn https://e3nn.org/ · tutorial https://blondegeek.github.io/e3nn_tutorial/ · QM9 https://quantum-machine.org/datasets/

**Deliverable:** an E(3)-equivariant GNN predicting a QM9 property; ablate equivariance against a non-equivariant baseline and report the **data-efficiency** gain (accuracy at 1k / 10k / full labels).
**Whiteboard checkpoint:** define invariance vs equivariance formally (f(g·x)=g·f(x)); show why 3D-equivariance matters for energies/forces; explain how EGNN updates coordinates equivariantly without spherical harmonics; state when a 2D-graph model beats a 3D one (and vice versa).

**5A CAPSTONE (folded into the weeks, ~10 hrs of the 90) — "the representation ladder":** on one public ADMET/QM9 task, compare **fingerprint+XGBoost vs D-MPNN (Chemprop) vs 3D-equivariant GNN** on the same scaffold split; report accuracy, data-efficiency, and compute cost; include your from-scratch message-passing implementation. Public repo + short write-up. *This single artifact is the strongest proof of molecular-ML competence you can show a panel.*

---

## 5B — Molecular Foundation Models & Self-Supervised Pretraining · ~50 hrs · PRIORITY: medium-high
*Low-data is the norm in drug discovery; pretraining is how you win it. Also directly on the small-molecule squad's foundation-model thrust.*

**Goal:** pretrain-then-finetune molecular models and know when SSL pays off. By end you can run a small masked-SMILES/graph pretrain, finetune it, and quantify the transfer gain vs training from scratch.

### Week 4 — SSL objectives + SMILES/graph pretraining (~25 hrs)
- **Read the landscape (SSL for molecules):** ChemBERTa (masked-SMILES BERT), Grover (graph SSL), MolCLR (graph contrastive), Uni-Mol (3D-aware). Read → tabulate objective, representation, and what each pretraining signal captures. — **~10 hrs** — ChemBERTa https://arxiv.org/abs/2010.09885 · MolCLR https://github.com/yuyangw/MolCLR · Uni-Mol https://github.com/deepmodeling/Uni-Mol · MolFormer https://github.com/IBM/molformer
- **Run a small masked-SMILES pretrain** on a public unlabeled set (e.g., a ZINC/ChEMBL SMILES subset) with a HF encoder — masking, tokenization, the MLM objective. — **~15 hrs** — HF course (for the mechanics) https://huggingface.co/learn/llm-course · ChEMBL https://www.ebi.ac.uk/chembl/ · ZINC https://zinc.docking.org/

**Deliverable:** a working masked-SMILES pretraining run (loss curve + a sanity-check on masked-token recovery).
**Whiteboard checkpoint:** write the masked-language-modeling objective; explain what a *valid* molecular augmentation is for contrastive SSL (and why bad augmentations break it); state the difference between what a SMILES-LM vs a 3D-SSL model learns.

### Week 5 — Finetune + transfer study (~25 hrs)
- **Finetune a pretrained molecular encoder** (your Week-4 model, or a public checkpoint) on a small TDC task; compare vs your 5A from-scratch model. — **~15 hrs** — TDC https://tdcommons.ai/
- **Transfer/data-ablation study** — measure the transfer gain at 100 / 1k / 10k labels; identify where pretraining helps vs where it's neutral. — **~10 hrs**

**Deliverable (5B capstone):** a pretrain→finetune notebook showing the transfer gain vs from-scratch across label-budget regimes, with an honest "when it didn't help" section.
**Whiteboard checkpoint:** when does SSL pretraining help vs hurt a downstream molecular task? Why can a fingerprint+GBM still beat a fine-tuned foundation model on a small, well-defined endpoint?

---

## 5C — Generative Chemistry: GFlowNets + Equivariant Diffusion · ~70 hrs · PRIORITY: high (closes the "generative molecules" gap)
*This is the DESRES wish-list item your resume can't yet claim. Build it and it becomes an honest, defensible bullet.*

**Goal:** generate diverse, high-reward, valid molecules; understand GFlowNets (flow-matching) vs RL vs VAE/flow vs diffusion, cold; run a property-reward generator and evaluate it properly.

### Week 6 — Generative baselines + the landscape + metrics (~25 hrs)
- **VAE/flow baselines** — skim JT-VAE and MoFlow; understand latent-space generation and its failure modes (validity, posterior collapse). — **~8 hrs** — JT-VAE https://arxiv.org/abs/1802.04364 · MoFlow https://arxiv.org/abs/2006.10137
- **The generative-model landscape for molecules** — a survey pass: search-based vs autoregressive vs latent vs diffusion vs GFlowNet; the axes that matter (diversity, validity, novelty, optimizability). — **~7 hrs** — DeepChem generative tutorials https://deepchem.io/tutorials/
- **Metrics + reward design** — validity, uniqueness, novelty, diversity (internal distance), and property rewards (QED, SA-score, a docking/predicted-activity proxy); GuacaMol/MOSES benchmarks. — **~10 hrs** — MOSES https://github.com/molecularsets/moses · GuacaMol https://github.com/BenevolentAI/guacamol

**Deliverable:** a metrics + reward module (validity/novelty/diversity + a property reward) reusable by the generators below.
**Whiteboard checkpoint:** define validity/uniqueness/novelty/diversity precisely; explain why "high reward" alone is a bad objective (mode collapse) and why diversity matters in hit-finding.

### Week 7 — GFlowNets (~25 hrs)
- **GFlowNet theory** — Bengio et al. "Flow Network based Generative Models" + "GFlowNet Foundations"; understand the flow-matching / trajectory-balance objective and the sample-diversity-∝-reward guarantee. — **~12 hrs** — https://arxiv.org/abs/2106.04399 · foundations https://arxiv.org/abs/2111.09266
- **GFlowNet hands-on** — work the tutorial/code; train a GFlowNet on a small fragment- or atom-based molecule-building environment with your Week-6 property reward. — **~13 hrs** — https://github.com/GFNOrg/gflownet

**Deliverable:** a GFlowNet molecule generator with a property reward; report diversity/validity/novelty/reward vs a naive RL or random baseline.
**Whiteboard checkpoint:** write the trajectory-balance (or flow-matching) objective; explain **why a GFlowNet samples proportionally to reward and therefore stays diverse**, whereas reward-maximizing RL collapses to a few modes; explain the DAG-of-states construction for molecule building.

### Week 8 — Equivariant diffusion for 3D molecule generation (~20 hrs)
- **EDM (equivariant diffusion) + GeoDiff** — read; understand denoising over 3D coordinates with E(3)-equivariance; connect to torsional diffusion (conformers, from 5D). — **~12 hrs** — EDM https://github.com/ehoogeboom/e3_diffusion_for_molecules · GeoDiff https://github.com/MinkaiXu/GeoDiff
- **Run a small 3D generation** (pretrained or tiny-train) and evaluate geometry validity. — **~8 hrs**

**Deliverable (5C capstone):** the GFlowNet (or diffusion) generator + evaluation, as a public repo; a note comparing GFlowNet vs RL vs diffusion trade-offs for molecular design.
**Whiteboard checkpoint:** how does equivariant diffusion denoise 3D coordinates while respecting rotation/translation symmetry? When would you choose a 2D-graph GFlowNet over a 3D diffusion model for de-novo design?

---

## 5D — Physics-Based Structure & Simulation · ~110 hrs · PRIORITY: high (the DESRES core; bridges Phase Q)
*The "more accurate approaches for molecular simulation" leg. Owning the accuracy/cost ladder is what a computational-chemistry panel probes hardest.*

**Goal:** run and reason about conformer generation, docking, free-energy, and MD, and place every method on the ladder MMFF → semi-empirical (GFN2-xTB) → ML-potential → DFT → FEP → MD. Reuses your DFT/quantum-chemistry strength.

### Week 9 — Conformer generation & 3D structural analysis (~30 hrs)
Follow the standalone spec end-to-end (it is written for exactly this slot).
- **Conformer pipeline** — RDKit ETKDGv3 + MMFF94s optimization + Butina RMSD clustering; **COV/MAT/RMSD** benchmarking against **GEOM-Drugs** reference ensembles; **DFT re-ranking** of top conformers (Psi4/PySCF, ωB97X-D/def2-SVP) reusing your quantum-chem skill. — **~30 hrs** — RDKit https://www.rdkit.org/ · GEOM https://github.com/learningmatter-mit/geom · Psi4/PySCF · (ML potentials MACE-OFF/TorchANI for the accuracy-vs-cost curve)

**Deliverable:** the conformer-generation + 3D-analysis repo (COV/MAT/RMSD + energy-ranking correlation MMFF-vs-DFT).
**Whiteboard checkpoint:** why is MMFF energy ordering unreliable and what fixes it? define COV vs MAT (recall vs precision); why symmetry-corrected best-RMSD; why def2-SVP + a dispersion correction.

### Week 10 — Molecular docking (~25 hrs)
- **Classical docking** — AutoDock Vina / smina: search + scoring; prepare a public target (from PDBbind or DUD-E), dock a ligand set, evaluate pose RMSD and screening enrichment (EF/BEDROC). — **~13 hrs** — Vina https://vina.scripps.edu/ · PDBbind https://www.pdbbind-plus.org.cn/ · DUD-E http://dude.docking.org/
- **ML docking / scoring** — gnina (CNN scoring) and DiffDock (generative pose prediction); compare to classical docking on the same target. — **~12 hrs** — gnina https://github.com/gnina/gnina · DiffDock https://github.com/gcorso/DiffDock

**Deliverable:** a docking mini-study on a public target — classical vs ML docking, pose RMSD + enrichment.
**Whiteboard checkpoint:** what does a docking score approximate and why is it a poor absolute affinity? why is pose RMSD not enough without enrichment? what does DiffDock change vs search-based docking?

### Week 11 — Free energy (FEP / ABFEP) (~25 hrs)
- **Statistical-mechanics intuition** — free energy, ensembles, thermodynamic cycles, why relative FEP cancels errors; alchemistry primer. — **~12 hrs** — https://www.alchemistry.org/wiki/Main_Page
- **Hands-on relative FEP** — OpenFE: set up a small relative binding free-energy calculation between two congeneric ligands; read the result critically. — **~13 hrs** — OpenFE https://openfree.energy/

**Deliverable:** a documented relative-FEP example + a one-page "when is FEP worth the cost" note.
**Whiteboard checkpoint:** draw the thermodynamic cycle behind relative binding FEP and explain what cancels; why FEP is accurate but expensive; where it fails (force-field quality, sampling, large perturbations).

### Week 12 — Molecular dynamics (~30 hrs)
- **OpenMM basics** — force fields, integrators, thermostats/barostats, periodic boundaries; run a short protein–ligand MD; the "Making it Rain" colabs are the fastest on-ramp. — **~18 hrs** — OpenMM https://openmm.org/ · Making-it-Rain https://github.com/pablo-arantes/Making-it-rain
- **Trajectory analysis + ML potentials** — MDAnalysis (RMSD/RMSF/contacts); ML interatomic potentials (MACE-OFF / ANI-2x) as fast DFT-quality surrogates and their transferability limits. — **~12 hrs** — MDAnalysis https://www.mdanalysis.org/

**Deliverable (5D CAPSTONE):** a conformer→docking pipeline on a public target (pose/enrichment) + a short OpenMM MD run with trajectory analysis + the FEP exercise; a write-up placing **every method on the accuracy/cost ladder** with your quantum-chem work at the top.
**Whiteboard checkpoint:** the full accuracy/cost ladder and where each rung breaks; what a force field is and its limits; why MD needs thermostats/barostats; how an ML potential reaches DFT-quality at MM cost and where it fails; how this connects to your VQE/DMET quantum work.

---

## 5E — Uncertainty Quantification & Probabilistic ML · ~55 hrs · PRIORITY: high (cross-cutting; a stated squad priority)
*A prediction without calibrated uncertainty is not decision-grade. This is also a direct squad deliverable and a strong interview differentiator.*

**Goal:** put valid, calibrated uncertainty on molecular predictions; know conformal prediction, GPs/SVGP, calibration, and ensembles/MC-dropout cold, and when to use each.

### Week 13 — Conformal prediction + calibration (~28 hrs)
- **Conformal prediction** — Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction"; split/inductive conformal; how it gives distribution-free marginal coverage. — **~14 hrs** — https://arxiv.org/abs/2107.07511
- **MAPIE hands-on** — add conformal prediction intervals to your 5A property model on a public ADMET set; verify empirical coverage at target α. — **~8 hrs** — https://github.com/scikit-learn-contrib/MAPIE
- **Calibration** — ECE, reliability diagrams, temperature scaling; what calibration does and doesn't tell you. — **~6 hrs**

**Deliverable:** conformal intervals on your 5A model with an empirical-coverage plot (does 90% nominal give ~90% coverage?).
**Whiteboard checkpoint:** why split conformal gives valid marginal coverage without distributional assumptions; the exchangeability assumption and how scaffold/temporal shift breaks it; what ECE measures and its failure modes.

### Week 14 — Gaussian processes / SVGP + the Bayesian view (~27 hrs)
- **GP regression** — kernels, the posterior mean/variance, what the kernel encodes; the intuition before the algebra. — **~10 hrs** — Murphy *Probabilistic ML* https://probml.github.io/pml-book/ · (Rasmussen & Williams GPML as reference)
- **Sparse variational GP (SVGP) hands-on** — GPyTorch: an SVGP on a public ADMET set for scalable calibrated regression. — **~10 hrs** — GPyTorch https://gpytorch.ai/
- **Compare the UQ toolbox** — conformal vs GP vs MC-dropout vs deep ensembles on the same task; honest trade-offs. — **~7 hrs**

**Deliverable (5E CAPSTONE):** add **both** conformal intervals and an **SVGP** to your 5A model; report empirical coverage + calibration; a four-way UQ comparison (conformal / GP / MC-dropout / deep-ensemble) with a recommendation.
**Whiteboard checkpoint:** the GP posterior mean/variance formulas and what the kernel encodes; why deep ensembles are often the strongest practical UQ; conformal vs Bayesian coverage guarantees; when you'd ship which method.

---

## 5F — Applied Cheminformatics & Discovery ML · ~75 hrs · PRIORITY: medium-high
*The applied methods a small-molecule squad ships. Pick a lane for the capstone based on the role you're closest to landing.*

**Goal:** own retrosynthesis/reaction-condition, DEL modeling, multitask/MMoE, contrastive representation, federated learning, and active-learning virtual screening well enough to contribute and to defend in a panel.

### Week 15 — Retrosynthesis + reaction-condition (~25 hrs)
- **Retrosynthesis (AiZynthFinder)** — MCTS over a building-block stock with a template/expansion policy; run it on public targets; understand the search. — **~13 hrs** — https://github.com/MolecularAI/aizynthfinder
- **Reaction representation + condition prediction** — reaction fingerprints/graphs; predict conditions (reagent/solvent/temp) on public reaction data (USPTO / Open Reaction Database). — **~12 hrs** — Open Reaction Database https://open-reaction-database.org/

**Deliverable:** a retrosynthesis run + a small reaction-condition model with honest top-k accuracy.
**Whiteboard checkpoint:** how MCTS retrosynthesis searches (selection/expansion/rollout/backup); why synthesizability matters as a generative constraint; why reaction-condition prediction is multi-label and noisy.

### Week 16 — DEL, multitask/contrastive, federated + virtual screening (~50 hrs)
- **DEL (DNA-encoded library) modeling** — enrichment/denoising from noisy barcoded counts; the public **BELKA** dataset. — **~15 hrs** — https://www.kaggle.com/competitions/leash-BELKA
- **Multitask / MMoE + contrastive** — a multi-endpoint ADMET model (shared trunk, per-task heads); Mixture-of-Experts to reduce negative transfer; contrastive representation of binding/affinity. — **~15 hrs** — (build on your 5A stack + TDC multi-endpoint tasks https://tdcommons.ai/)
- **Federated learning + active-learning virtual screening** — Flower: a federated multi-endpoint ADMET demo (train across simulated sites without pooling data); an active-learning loop that docks + trains an ML surrogate over a library (ties 5A + 5D + 5E). — **~20 hrs** — Flower https://flower.ai/

**Deliverable (5F CAPSTONE — pick ONE lane, ship a public repo + write-up):**
1. **Agentic decision-support:** wrap your 5A property + 5E UQ model as an **MCP tool** with a synthesizability check — a "drug-hunter co-pilot" (ties to Phase 6 and mirrors your critic-agent doc-review project's patterns).
2. **Reaction/retrosynthesis pipeline** (Week-15 work, hardened).
3. **Federated multi-endpoint ADMET** demo.
4. **DEL enrichment** model on BELKA.
**Whiteboard checkpoint:** why DEL counts are noisy and how enrichment modeling denoises them; when multitask helps vs negative transfer and how MMoE mitigates it; what federated learning trades off (privacy vs communication/heterogeneity); how active learning chooses the next compounds to screen.

---

## PHASE 5 — TRACK INTERVIEW SELF-TEST (must pass cold, no notes → Tier 2)
Answer each out loud, whiteboard where relevant:
1. **Representations:** message passing vs convolution; what a D-MPNN buys over a fingerprint and when the fingerprint+GBM still wins; the WL expressiveness bound; over-smoothing and fixes.
2. **3D/equivariance:** invariance vs equivariance formally; why 3D-equivariance matters for energies/forces; how EGNN updates coordinates equivariantly; when 2D beats 3D.
3. **Pretraining:** the MLM objective; a valid molecular augmentation for contrastive SSL; when SSL helps vs hurts.
4. **Generative:** the GFlowNet objective and **why it yields reward-proportional diversity** (vs RL mode-collapse); how equivariant diffusion respects symmetry; validity/novelty/diversity metrics.
5. **Physics:** the full MMFF→xTB→ML-potential→DFT→FEP→MD **accuracy/cost ladder** and where each breaks; the thermodynamic cycle behind relative FEP; what a docking score really approximates; how your VQE/DMET work sits at the top of the ladder.
6. **Uncertainty:** why split conformal gives valid coverage without distributional assumptions and how molecular distribution-shift breaks exchangeability; GP posterior + what the kernel encodes; why deep ensembles are often best in practice; what ECE measures.
7. **Applied:** MCTS retrosynthesis; DEL count denoising; multitask vs negative transfer (MMoE); federated-learning trade-offs; the active-learning virtual-screening loop.
8. **System design (ties Phase 4/9):** design a molecular-property prediction service with calibrated UQ and scaffold-aware validation; design an active-learning virtual-screening pipeline over a billion-compound library.

**Flagship artifacts this phase produces (for the resume/portfolio, all public):** (5A) the representation-ladder study; (5C) a GFlowNet/diffusion generator; (5D) the conformer→docking→FEP→MD physics repo; (5E) the four-way UQ comparison; (5F) one applied capstone. Any one of these is a legitimate MLSB / LoG / AI4Science workshop submission — aim at least one there (Thread R).

---

## PHASE Q — Quantum Computing + Quantum Machine Learning (full-depth track) · ~305 hrs (+ Capstone Q ~30) · *off the job-critical path; concentrated after Phase 8, ~Feb–Apr 2028*
**Goal:** match hands-on VQE/ADAPT-VQE/DMET-VQE work with rigorous theory — derive VQE from the variational principle, explain barren plateaus mathematically, defend ansatz/measurement choices, and discuss QML's real limits honestly (Schuld). Depth on top of experience, off the job-critical path so it never delays the ladder; it also **grounds 5D's physics** (why DFT/FEP are accurate, the classical↔quantum interface).
> **COI:** public molecules only (H₂/LiH/H₂O or a QM9 subset); avoid your Lilly molecules (no keto-enol/HATU specifics) and anything from the vendor engagement.

**Intuition-first entry (before the rigorous courses):** Quantum Country (spaced-repetition primer) https://quantum.country/qcvc · MIT 8.04 (Adams) https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2013/video_galleries/lecture-videos/ — **~10 hrs**

### Q.1 — Group theory & symmetry (~22 hrs)
- Group theory for physicists (lectures) — **~14 hrs** — https://www.youtube.com/results?search_query=group+theory+for+physicists+lectures
- GDL lectures for the Lie-group/equivariance link (shared with 5A) — **~8 hrs** — https://geometricdeeplearning.com/lectures/
*Checkpoint:* groups/representations; why symmetry underlies both quantum operators and equivariant nets.

### Q.2 — Quantum-mechanics math (Hilbert spaces, operators, tensor products) (~25 hrs)
- MIT 8.04 (L6–12) — **~15 hrs** — https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2013/ · Quantum Country https://quantum.country/qcvc — **~6 hrs** · (optional) Schuller "Geometric Anatomy" sampled — **~4 hrs** https://www.youtube.com/playlist?list=PLPH7f_7ZlzxTi6kS4vCmv4ZKm9u8g5yic
*Whiteboard:* define a Hilbert space; tensor-product two qubit states; unitarity, Hermitian operators, why measurement outcomes are eigenvalues; pure vs mixed states + the density matrix.

### Q.3 — Quantum computing fundamentals: the Watrous / IBM track (~80 hrs) · rigorous core
- **Basics of Quantum Information** — **~25 hrs** — https://learning.quantum.ibm.com/course/basics-of-quantum-information · YouTube https://www.youtube.com/playlist?list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR
- **Fundamentals of Quantum Algorithms** — **~25 hrs** — https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms
- **General Formulation** (density matrices, channels, POVMs — matters for noise/error mitigation) — **~18 hrs** — https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information
- **Foundations of Quantum Error Correction** — **~12 hrs** — https://learning.quantum.ibm.com/course/foundations-of-quantum-error-correction
- Companion: Nielsen & Chuang ch 1–4, 8, 10 (read alongside).
*Whiteboard:* single/multi-qubit gates; entanglement; the standard algorithms (Deutsch-Jozsa → phase estimation) and what each teaches.

### Q.4 — Variational algorithms (VQE/ADAPT/DMET/QAOA) (~45 hrs)
- PennyLane chemistry demos (run the derivations behind code you've used) — **~15 hrs** — https://pennylane.ai/qml/demos_quantum-chemistry · PennyLane Codebook (VQAs) https://pennylane.ai/codebook
- VQE (Peruzzo 2014) https://www.nature.com/articles/ncomms5213 · ADAPT-VQE (Grimsley 2019) https://www.nature.com/articles/s41467-019-10988-2 · DMET-VQE survey https://arxiv.org/abs/2108.08987 — **~18 hrs**
- QAOA (Farhi 2014) https://arxiv.org/abs/1411.4028 · Hadfield review https://arxiv.org/abs/1709.03489 · Pauli grouping / measurement (shot-budget) https://arxiv.org/abs/1907.13117 — **~12 hrs**
*Whiteboard:* derive the variational principle; VQE end-to-end (ansatz, parameter-shift rule, optimizer); why ADAPT-VQE is parameter-efficient; when DMET fragmentation is worth the overhead.

### Q.5 — QML theory (the honest-take material) (~42 hrs)
- PennyLane QML topic page (Schuld-curated) — **~10 hrs** — https://pennylane.ai/topics/quantum-machine-learning
- Schuld "Taking stock of QML: a critical perspective" (the talk every senior should watch) — **~6 hrs** — https://www.youtube.com/results?search_query=maria+schuld+taking+stock+quantum+machine+learning
- Schuld & Petruccione ch 4–7 · Key papers: QML=kernels https://arxiv.org/abs/2101.11020 · Barren plateaus https://www.nature.com/articles/s41467-018-07090-4 · Expressibility https://arxiv.org/abs/1905.10876 · "Is quantum advantage the right goal?" https://arxiv.org/abs/2203.01340 · PQC encodings https://arxiv.org/abs/2008.08605 — **~26 hrs** · courses to pull from: Péré https://github.com/Christophe-pere/QML-Course · Hjorth-Jensen https://github.com/CompPhysics/QuantumComputingMachineLearning
*Whiteboard:* barren plateaus mathematically + mitigations; QML-as-kernel-methods; the honest hype-vs-promise take (cite Schuld).

### Q.6 — Quantum chemistry on quantum computers (~60 hrs)
- Qiskit Global Summer School (chemistry + variational tracks) — **~20 hrs** — https://www.youtube.com/@qiskit/playlists
- IBM Quantum Learning + Sample-based/Krylov Quantum Diagonalization (SQD) — **~25 hrs** — https://learning.quantum.ibm.com/ · https://www.ibm.com/quantum/blog/iql-migration
- Build/replicate small end-to-end chemistry pipelines (public molecules) — **~15 hrs**

**CAPSTONE Q (~30 hrs) · public molecules:** H₂/LiH/H₂O or a QM9 subset — vanilla VQE + UCCSD, ADAPT-VQE, DMET-VQE, benchmarked vs classical CCSD/6-31G; 4-page write-up defending ansatz/optimizer/measurement grouping/noise mitigation/basis set + a barren-plateau analysis; optional delta-ML surrogate connecting to the 5A/5D capstones.
**Phase Q self-test:** derive VQE from the variational principle; the parameter-shift rule and why it works; a barren-plateau argument; VQE vs QAOA — when each and when no quantum algorithm; honest QML take (Schuld); DMET fragmentation and the classical-quantum interface. **→ QML career branch opens.**

---

## PHASE 6 — Generative & Agentic AI (scientific / chemistry agents) · ~155 hrs (+ Capstone 4) · *mid-Oct – late Nov 2027*
**Goal:** own modern LLM adaptation, multimodal, and agentic systems — reframed toward science. By end you can wrap a model as an MCP tool, build a multi-step research/critique agent, and evaluate agents rigorously (directly reinforcing your critic-agent doc-review project).

### Week 1–2 — Advanced LLMs / adaptation (~42 hrs)
- LoRA https://arxiv.org/abs/2106.09685 · QLoRA https://arxiv.org/abs/2305.14314 · Raschka (fine-tuning) https://magazine.sebastianraschka.com/ — **~18 hrs**
- Alignment: RLHF https://arxiv.org/abs/2203.02155 · DPO https://arxiv.org/abs/2305.18290 — **~10 hrs**
- HF LLM course (adaptation units) + Chip Huyen *AI Engineering* — **~14 hrs** — https://huggingface.co/learn/llm-course
*Checkpoint:* LoRA/QLoRA mechanics + when to use; RLHF vs DPO; parameter-efficient fine-tuning trade-offs.

### Week 3 — Multimodal / VLM (~26 hrs)
- CLIP https://arxiv.org/abs/2103.00020 · LLaVA https://arxiv.org/abs/2304.08485 · Flamingo https://arxiv.org/abs/2204.14198 — **~18 hrs** (read + run a small CLIP retrieval)
- Relevance to your work: multimodal for imaging+text and molecular-graph+text — **~8 hrs**
*Checkpoint:* how CLIP aligns modalities; where a VLM helps (and its failure modes) in scientific data.

### Week 4–5 — Agents, MCP & evaluation (~52 hrs) — *reframed to scientific/chemistry agents; mirrors your project*
- HF Agents course + HF MCP course — **~18 hrs** — https://huggingface.co/learn/agents-course · https://huggingface.co/learn/mcp-course
- Anthropic — Building Effective Agents (patterns: routing, orchestrator-worker, evaluator-optimizer = the **critic loop** in your project) — **~8 hrs** — https://www.anthropic.com/research/building-effective-agents
- LangGraph (stateful agent graphs) + Andrew Ng short courses — **~12 hrs** — https://langchain-ai.github.io/langgraph/ · https://www.deeplearning.ai/short-courses/
- **Agent/critique evaluation** — how to evaluate an agent and a critic honestly (task success, faithfulness, self-consistency) — **~14 hrs** — (Eugene Yan evals https://eugeneyan.com/writing/llm-evaluators/)

**CAPSTONE 4 · public data:** an **MCP-based scientific assistant** — a multi-step agent (RAG + tools + memory + a **critic/verifier loop** + eval) over open literature, exposing a molecular-property/UQ tool (ties to 5E/5F). Sources: PMC-OA https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ · arXiv https://arxiv.org/ · PubMed E-utilities https://www.ncbi.nlm.nih.gov/books/NBK25501/ · Semantic Scholar API https://www.semanticscholar.org/product/api. *Never ingest internal data.*
**Phase 6 self-test:** design an agent with a critic/verifier loop and explain how you'd evaluate the critic; MCP tool-exposure end-to-end; RLHF vs DPO; when an agent beats a single well-prompted call (and when it doesn't).

---

## PHASE 7 — Computer Vision Expert (classical / geometric / 3D / edge) · ~175 hrs (+ Capstone 5) · *late Nov 2027 – early Jan 2028*
**Goal:** round out into a broad CV expert so no vision question catches you flat (video already covered in Phase 2V), and gain real deployment/edge skill. By end you can reason about multi-view geometry, port a model to an optimized C++/TensorRT pipeline, and speak to 3D/neural rendering.

### Week 1–2 — Classical & geometric CV (~45 hrs)
- **First Principles of Computer Vision (Nayar)** — **~18 hrs** — https://fpcv.cs.columbia.edu/ · Coursera https://www.coursera.org/specializations/firstprinciplesofcomputervision
- **Cyrill Stachniss** (photogrammetry/SLAM lectures) — **~12 hrs** — https://www.youtube.com/@CyrillStachniss
- **Szeliski (free book)** as reference; **COLMAP / OpenMVG / ORB-SLAM3 / Open3D** hands-on — **~15 hrs** — https://szeliski.org/Book/ · https://colmap.github.io/ · https://github.com/openMVG/openMVG · https://github.com/UZ-SLAMLab/ORB_SLAM3 · http://www.open3d.org/
*Whiteboard:* pinhole model + epipolar constraint; SIFT & scale/rotation invariance; SfM pipeline.

### Week 3 — Modern DL vision breadth (~35 hrs)
- **Michigan EECS 498 (Justin Johnson)** — **~20 hrs** — https://web.eecs.umich.edu/~justincj/teaching/eecs498/ · playlist https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r
- Detection frameworks — Detectron2 https://github.com/facebookresearch/detectron2 · MMDetection https://github.com/open-mmlab/mmdetection · Ultralytics YOLO https://docs.ultralytics.com/ — **~15 hrs**
*Whiteboard:* two-stage vs one-stage detectors; anchor-based vs anchor-free; NMS.

### Week 4 — Deployment & optimization for hardware/edge (~50 hrs)
- OpenCV https://docs.opencv.org/ · PyImageSearch https://pyimagesearch.com/ — **~12 hrs**
- **ONNX + TensorRT** (export → optimize → quantize) — **~20 hrs** — https://onnx.ai/ · https://developer.nvidia.com/tensorrt
- **CUDA + PMPP (Kirk & Hwu)** + NVIDIA DLI; Triton / DeepStream — **~18 hrs** — https://docs.nvidia.com/cuda/cuda-c-programming-guide/ · https://www.nvidia.com/en-us/training/ · https://github.com/triton-inference-server/server · https://developer.nvidia.com/deepstream-sdk
*Whiteboard:* what TensorRT does; INT8 calibration; porting a PyTorch CV model to a real-time C++ pipeline.

### Week 5 — 3D vision & neural rendering (~30 hrs) + Capstone 5
- **NeRF** https://arxiv.org/abs/2003.08934 · **3D Gaussian Splatting** https://arxiv.org/abs/2308.04079 · nerfstudio https://docs.nerf.studio/ · Open3D point clouds http://www.open3d.org/ — **~30 hrs**

**CAPSTONE 5 (deployment) · public data:** SfM/calibration/visual-odometry on public sets, **or** take Capstone-1/2's model → optimized C++ inference (TensorRT/ONNX, quantized) with a latency/accuracy report. Datasets: KITTI https://www.cvlibs.net/datasets/kitti/ · TUM RGB-D https://cvg.cit.tum.de/data/datasets/rgbd-dataset · ETH3D https://www.eth3d.net/
**Phase 7 self-test:** pinhole + epipolar; SIFT invariance; two-stage vs one-stage; what TensorRT does; port a PyTorch CV model to a real-time C++ pipeline.

---

## PHASE 8 — Research + Big-Tech Interview Crescendo · ~160 hrs · *Jan – mid-Feb 2028* · → Tier 3
**Goal:** convert the whole plan into offers. RE loop = 2 medium-hard coding + ML fundamentals + ML system design + behavioral (+ a research discussion for research roles). Research loops **ban AI tools** — rehearse unaided.

### Week 1–2 — DSA at RE weight (~55 hrs)
- Timed **NeetCode 150 → Blind 75 → company-tagged** — **~55 hrs** — https://neetcode.io/ · https://leetcode.com/ · DesignGurus https://www.designgurus.io/ · Striver https://takeuforward.org/. Blank editor, ~35 min, code must run; ~4–6/day.
*Checkpoint:* a random LeetCode hard in ~45 min, clean running code.

### Week 3 — Implement-by-hand ML (~35 hrs) · highest value
- Timed, unaided, blank file: attention, a loss (your domain loss), a sampler, a diffusion step, a training loop, backprop, **a GNN layer** — plus an ML-math/theory sweep (gradients, KL, sampling, convergence, "L1 sparsity via gradients"). Draws on Thread M all year.
*Checkpoint:* implement attention + a loss + a sampler + a GNN layer unaided, timed.

### Week 4 — ML system design (~35 hrs)
- ML system design first (Chip Huyen); then generic system design for loops that include it — ByteByteGo https://bytebytego.com/ · Gaurav Sen https://www.youtube.com/@gkcs · DDIA + MIT 6.824 https://pdos.csail.mit.edu/6.824/. 8–10 prompts, recorded, self-critiqued — including **a molecular-property serving system, a virtual-screening pipeline, and an imaging pipeline with domain-shift handling.**
*Checkpoint:* a full ML-system-design whiteboard in 45 min.

### Week 5 — Research job-talk + behavioral (~35 hrs)
- A 20-min talk + Q&A on your **two strongest capstones** (the imaging flagship + the molecular/physics capstone); ~6–8 STAR stories per org; low-level/OO design where the loop includes it; live mocks (Pramp / interviewing.io).
**Phase 8 self-test:** LeetCode hard in ~45 min clean; implement attention + a loss + a sampler + a GNN layer unaided/timed; a 45-min system-design whiteboard; a 20-min job-talk; four STAR stories from memory. **→ Tier 3 top-lab loops. Intensive interviewing.**

---

## 7. Master Calendar (sequential, from Sep 1, 2026, at ~30 h/wk structured; self-tests are the true gates)

| Weeks | Period | Phase (calendar weeks) | Milestone |
|---|---|---|---|
| 0 | Sep 1–7, 2026 | bio-clock + new-project ramp; setup | schedule locked |
| 1–3 | Sep 2026 | Phase 0 (2) | Phase-0 self-test → **start applying (Tier 0)** |
| 3–13 | late Sep – early Dec 2026 | Phase 1 (10, incl. 2 consolidation) | Phase-1 self-test → **Threads begin** |
| 13–27 | Dec 2026 – mid-Mar 2027 | Phase 2 core + expansion + Capstone 1 (14) | **Tier 0/1 interview-ready** |
| 27–31 | mid-Mar – early Apr 2027 | Phase 2V + Capstone 2 (4) | **Tier 1** (imaging + phenomics → Recursion/PathAI) |
| 31–37 | April – mid-May 2027 | Phase 3 (6) | — |
| 37–43 | mid-May – late Jun 2027 | Phase 4 (6) | — |
| **43–58** | **late Jun – mid-Oct 2027** | **Molecular-ML Track 5A–5F (15)** | **Tier 2 — drug-discovery credible (DESRES/Isomorphic RE realistic)** |
| 58–64 | mid-Oct – late Nov 2027 | Phase 6 + Capstone 4 (6) | Tier 2 broadened |
| 64–70 | late Nov 2027 – early Jan 2028 | Phase 7 + Capstone 5 (6) | — |
| 70–76 | Jan – mid-Feb 2028 | **Phase 8 interview crescendo (6)** | **Tier 3 loops — DeepMind / Isomorphic / DESRES.** Intensive interviewing |
| 76–87 | ~Feb – Apr 2028 (H1 2028) | **Phase Q full + Capstone Q (11)** — concentrated *after* the crescendo | QML branch opens; **full depth ~mid-2028** |

**Reading the calendar:** Threads T/M/R and the application ladder run **concurrently inside the weekly hours** (not extra calendar time), which is why the ~2,925 total hours fit ~87 calendar weeks. **Phase Q is deliberately off the job-critical path** — concentrated after Phase 8 so it never delays Tier-3 interviewing; if a QML-specific role appears earlier, pull Phase Q forward and interleave. Consolidation weeks and the weekend buffer are already absorbed into the durations above.

**Critical path to a DESRES/Isomorphic loop:** P0 → P1 → P3 → **5A + 5D (+ conformer capstone) + 5E** → P4 system-design slice → P8, Thread M throughout. 5C, Phase Q, and Phase 7 deepen the profile but aren't on that path — so a credible Tier-2 loop is reachable around the **end of Phase 5 (~mid-Oct 2027)**, with the full crescendo by **~mid-Feb 2028**.

## 8. Application ladder
- **Tier 0 (from Wk 3):** India medical-imaging AI — Qure.ai https://qure.ai/ · SigTuple https://sigtuple.com/ · GE HealthCare https://www.gehealthcare.com/ · NVIDIA https://www.nvidia.com/en-us/about-nvidia/careers/ · MSR India https://www.microsoft.com/en-us/research/lab/microsoft-research-india/
- **Tier 1 (+Capstone 2 + phenomics):** broader medical/biomedical imaging + video; **Recursion (phenomics — direct match)** https://www.recursion.com/careers · PathAI · Paige · Owkin https://www.owkin.com/ · Google Research India https://research.google/locations/india/
- **Tier 2 (+Molecular-ML core + conformer/docking):** **D. E. Shaw Research · Isomorphic Labs** https://www.isomorphiclabs.com/careers · **Google DeepMind** https://deepmind.google/about/careers/ · Insitro https://insitro.com/ · Genesis https://www.genesistherapeutics.ai/ · Iambic https://www.iambic.ai/ · Chai https://www.chaidiscovery.com/ · InstaDeep https://www.instadeep.com/ · Valence https://www.valencelabs.com/
- **Tier 3 (+interview crescendo):** full top-lab loops. **QML branch (+Capstone Q):** QpiAI https://www.qpiai.tech/careers · IBM Quantum https://www.ibm.com/quantum
- *Your internal small-molecule / discovery-imaging directions are natural internal moves this public skill-set prepares you for — via normal channels, portfolio kept public.*

## 9. Conflict-of-interest guardrails
Public data & generic methods only; personal hardware/accounts; **nothing proprietary or colleague-specific in any external artifact** (resume, public repos, interview talking points) — this includes the confidential squad briefings (their codenames/targets/Jira-IDs/names are for *your* skill-mapping only). Clear any public write-up through your org's external-publication review. Personal-hardware/public-data capstones aren't subject to the internal Artifactory rule; your Lilly work is.

## 10. Cross-cutting weekly habits
Paper-of-the-week (Evening) · Whiteboard-Friday concept (photo → prep deck) · Sunday teach-a-junior note · living glossary + war-chest. *(Re-implementation → Thread M; reproduce-a-paper → Thread R.)*

## 11. PhD-later decision guide (compact)
RS at frontier labs is PhD-gated; RE/Applied-Scientist generally isn't — RE is the near-term door, RS the post-PhD aspiration. **When:** after landing a strong RE role. **Where:** India external/part-time (IISc ERP https://www.iisc.ac.in/admissions/external-registration-programme-ph-d/), EU salaried/industrial (EURAXESS https://euraxess.ec.europa.eu/ — strongest funded route), US research-embedded. Set up now (zero cost): a capstone → workshop paper + preprint; 1–2 advisor relationships. Criteria: advisor fit → topic continuity → funding → part-time feasibility → location/visa → prestige (last).

## 12. Publication & portfolio strategy (compact)
1–2 workshop papers + 1 preprint + clean public repos over the window. Imaging → MIDL/MICCAI workshop; molecular → MLSB/LoG. A merged PR into **e3nn / Chemprop / MONAI / DeepChem / PennyLane** beats a blog post. Reproduce-a-paper (Thread R) = referral hook + level-proof.

## 13. Risk register (compact)
Confusion compounding → the Confusion Buffer (understanding-gated self-tests, spaced re-derivation, triangulation). **Overload/burnout (the #1 risk at 40+ h/wk on top of work)** → sleep protection, weekend buffer, rest day, consolidation weeks; use them early. Breadth-without-depth → self-tests gate; Thread M enforces depth; the molecular track stays the priority. QML off the job-critical path. Everything portfolio-facing stays public and defensible.

---

# 14. RESOURCES — unified index (fused: core + all additions, organized by topic)
*One list per topic; earlier "v10-core / v11-additions" split removed. Everything a phase needs is together.*

**Math & foundations (P0/P1 + threaded math)** — StatQuest https://www.youtube.com/@statquest/playlists · https://statquest.org/ · 3B1B LA https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab · 3B1B Calc https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr · 3B1B NN https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi · MIT 18.06 https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/ · MIT 18.065 https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/ · Stat 110 https://projects.iq.harvard.edu/stat110/youtube · MIT RES.6-012 https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/ · Blitzstein & Hwang https://probabilitybook.net/ · Matrix Cookbook https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf · Matrix Calculus (Parr & Howard) https://explained.ai/matrix-calculus/ · Karpathy Z2H https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ · repo https://github.com/karpathy/nn-zero-to-hero · Boyd EE364A https://web.stanford.edu/~boyd/cvxbook/ · Ruder https://www.ruder.io/optimizing-gradient-descent/ · MacKay ITILA https://www.inference.org.uk/itila/book.html · Group theory https://www.youtube.com/results?search_query=group+theory+for+physicists+lectures · GDL https://geometricdeeplearning.com/lectures/ · Imperial LA https://www.coursera.org/learn/linear-algebra-machine-learning · Imperial Multivariate Calc https://www.coursera.org/learn/multivariate-calculus-machine-learning

**Biomedical imaging — core + expansion (P2)** — MRI Q&A https://mriquestions.com/index.html · OHBM https://www.youtube.com/@ohbmonline/playlists · Callaghan https://www.youtube.com/playlist?list=PLqEMVgX2VgZcG_xR1QRdsuc-X_zEgMx-Z · iBiology https://www.youtube.com/playlist?list=PLF513KEDjY9YBNhwMfX6jMl3PlpdW9TMP · AI for Medical Diagnosis https://www.coursera.org/learn/ai-for-medical-diagnosis · DigitalSreeni https://www.youtube.com/@DigitalSreeni/playlists · code https://github.com/bnsreenu/python_for_microscopists · CS231n https://cs231n.github.io/ · playlist https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv · Conv arithmetic https://arxiv.org/abs/1603.07285 · https://github.com/vdumoulin/conv_arithmetic · MONAI https://github.com/Project-MONAI/tutorials · https://www.youtube.com/@projectmonai/videos · nnU-Net https://github.com/MIC-DKFZ/nnUNet · Kitware https://www.kitware.com/developing-custom-3d-medical-image-segmentation-solutions-using-out-of-the-box-pipelines-in-monai/ · TorchIO https://torchio.readthedocs.io/ · Stanford AIMI https://aimi.stanford.edu/education/educational-resources · Metrics Reloaded https://www.nature.com/articles/s41592-023-02151-z · Yarin Gal uncertainty https://www.youtube.com/results?search_query=yarin+gal+uncertainty+deep+learning · **Papers:** U-Net https://arxiv.org/abs/1505.04597 · V-Net https://arxiv.org/abs/1606.04797 · Attention U-Net https://arxiv.org/abs/1804.03999 · nnU-Net https://www.nature.com/articles/s41592-020-01008-z · Loss survey https://arxiv.org/abs/2006.14822 · Generalized Dice https://arxiv.org/abs/1707.03237 · Focal Tversky https://arxiv.org/abs/1810.07842 · Boundary https://arxiv.org/abs/1812.07032 · ViT https://arxiv.org/abs/2010.11929 · TransUNet https://arxiv.org/abs/2102.04306 · Swin-UNet https://arxiv.org/abs/2105.05537 · UNETR https://arxiv.org/abs/2103.10504 · Swin UNETR https://arxiv.org/abs/2201.01266 · SAM https://arxiv.org/abs/2304.02643 · MedSAM https://www.nature.com/articles/s41467-024-44824-z · MedNeXt https://arxiv.org/abs/2303.09975 · **Expansion tools:** Cellpose https://www.cellpose.org/ · StarDist https://github.com/stardist/stardist · CellProfiler https://cellprofiler.org/ · Allen Brain Atlas https://atlas.brain-map.org/ · BrainGlobe https://brainglobe.info/ · ANTsPy https://github.com/ANTsX/ANTsPy · SimpleITK https://simpleitk.org/ · OpenSlide https://openslide.org/ · TIAToolbox https://github.com/TissueImageAnalytics/tiatoolbox · CLAM https://github.com/mahmoodlab/CLAM · QuPath https://qupath.github.io/ · napari https://napari.org/ · torchstain https://github.com/EIDOSLAB/torchstain · modAL https://github.com/modAL-python/modAL · DINOv2 https://github.com/facebookresearch/dinov2 · MAE https://arxiv.org/abs/2111.06377 · UNI https://github.com/mahmoodlab/UNI · Virchow (HF) https://huggingface.co/paige-ai · **Datasets:** Medical Decathlon http://medicaldecathlon.com/ · BraTS https://www.synapse.org/brats · TCIA https://www.cancerimagingarchive.net/ · IXI https://brain-development.org/ixi-dataset/ · LIVECell https://github.com/sartorius-research/LIVECell · Sartorius https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation · BBBC https://bbbc.broadinstitute.org/ · JUMP-CP https://jump-cellpainting.broadinstitute.org/ · RxRx https://www.rxrx.ai/

**Video / temporal (P2V)** — VideoMAE https://arxiv.org/abs/2203.12602 · TimeSformer https://arxiv.org/abs/2102.05095 · ViViT https://arxiv.org/abs/2103.15691 · SlowFast https://arxiv.org/abs/1812.03982 · SAM 2 https://arxiv.org/abs/2408.00714 · https://github.com/facebookresearch/sam2 · RAFT https://arxiv.org/abs/2003.12039 · ByteTrack https://arxiv.org/abs/2110.06864 · OC-SORT https://arxiv.org/abs/2203.14360 · BoT-SORT https://arxiv.org/abs/2206.14651 · HOTA https://arxiv.org/abs/2009.07736 · MOTChallenge https://motchallenge.net/ · DAVIS https://davischallenge.org/ · YouTube-VOS https://youtube-vos.org/ · DeepLabCut https://www.deeplabcut.org/ · SLEAP https://sleap.ai/ · TrackMate https://imagej.net/plugins/trackmate/ · Ultrack https://github.com/royerlab/ultrack · Cell Tracking Challenge http://celltrackingchallenge.net/ · CalMS21 https://data.caltech.edu/records/s0vdx-0k302 · MABe https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022

**DL theory & architectures (P3)** — CS25 https://web.stanford.edu/class/cs25/ · https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM · CS236 https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8 · Lilian Weng https://lilianweng.github.io/ · Annotated Transformer http://nlp.seas.harvard.edu/annotated-transformer/ · HF CV https://huggingface.co/learn/computer-vision-course · HF Diffusion https://huggingface.co/learn/diffusion-course · RoPE https://arxiv.org/abs/2104.09864 · RMSNorm https://arxiv.org/abs/1910.07467 · FlashAttention https://arxiv.org/abs/2205.14135 · GQA https://arxiv.org/abs/2305.13245 · SwiGLU https://arxiv.org/abs/2002.05202 · MoE/Switch https://arxiv.org/abs/2101.03961 · ConvNeXt https://arxiv.org/abs/2201.03545 · DINOv2 https://arxiv.org/abs/2304.07193 · DDPM https://arxiv.org/abs/2006.11239 · Yang Song score-based https://yang-song.net/blog/2021/score/ · Lilian Weng diffusion https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

**Production / MLOps / LLM systems (P4)** — CS336 https://stanford-cs336.github.io/spring2024/ · https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_ · HF NLP https://huggingface.co/learn/nlp-course · CS329S https://stanford-cs329s.github.io/ · Made With ML https://madewithml.com/ · Full Stack DL https://fullstackdeeplearning.com/course/2022/ · Chip Huyen https://huyenchip.com/ml-interviews-book/ · Hamel Husain https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/ · Eugene Yan patterns https://eugeneyan.com/writing/llm-patterns/ · evals https://eugeneyan.com/writing/llm-evaluators/ · Anthropic contextual retrieval https://www.anthropic.com/news/contextual-retrieval · Constitutional AI https://www.anthropic.com/news/constitutional-ai · LlamaIndex https://docs.llamaindex.ai/ · RAGAS https://docs.ragas.io/ · DSPy https://dspy.ai/ · Raschka https://magazine.sebastianraschka.com/ · HF multi-GPU https://huggingface.co/docs/transformers/en/perf_train_gpu_many · vLLM https://arxiv.org/abs/2309.06180 · ZeRO https://arxiv.org/abs/1910.02054

**Molecular ML & cheminformatics (Track 5A–5F)** — CS224W https://web.stanford.edu/class/cs224w/ · playlist https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn · Bronstein GDL https://geometricdeeplearning.com/lectures/ · proto-book https://arxiv.org/abs/2104.13478 · PyG https://pytorch-geometric.readthedocs.io/ · colabs https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html · DGL https://www.dgl.ai/ · DGL-LifeSci https://github.com/awslabs/dgl-lifesci · Chemprop https://github.com/chemprop/chemprop · DeepChem https://deepchem.io/tutorials/ · RDKit https://www.rdkit.org/ · Graphormer https://arxiv.org/abs/2106.05234 · GraphGPS https://arxiv.org/abs/2205.12454 · SchNet https://arxiv.org/abs/1706.08566 · DimeNet++ https://arxiv.org/abs/2011.14115 · GemNet https://arxiv.org/abs/2106.08903 · EGNN https://arxiv.org/abs/2102.09844 · e3nn https://e3nn.org/ · tutorial https://blondegeek.github.io/e3nn_tutorial/ · NequIP https://www.nature.com/articles/s41467-022-29939-5 · MACE https://arxiv.org/abs/2206.07697 · Allegro https://github.com/mir-group/allegro · Equiformer https://arxiv.org/abs/2206.11990 · MolFormer https://github.com/IBM/molformer · MolCLR https://github.com/yuyangw/MolCLR · ChemBERTa https://arxiv.org/abs/2010.09885 · Uni-Mol https://github.com/deepmodeling/Uni-Mol · **Generative:** GFlowNet https://arxiv.org/abs/2106.04399 · foundations https://arxiv.org/abs/2111.09266 · code https://github.com/GFNOrg/gflownet · EDM https://github.com/ehoogeboom/e3_diffusion_for_molecules · GeoDiff https://github.com/MinkaiXu/GeoDiff · Torsional Diffusion https://github.com/gcorso/torsional-diffusion · JT-VAE https://arxiv.org/abs/1802.04364 · MoFlow https://arxiv.org/abs/2006.10137 · MOSES https://github.com/molecularsets/moses · GuacaMol https://github.com/BenevolentAI/guacamol · **Physics:** GEOM https://github.com/learningmatter-mit/geom · AutoDock Vina https://vina.scripps.edu/ · gnina https://github.com/gnina/gnina · DiffDock https://github.com/gcorso/DiffDock · OpenMM https://openmm.org/ · OpenFE https://openfree.energy/ · MDAnalysis https://www.mdanalysis.org/ · Making-it-Rain https://github.com/pablo-arantes/Making-it-rain · alchemistry https://www.alchemistry.org/wiki/Main_Page · **UQ:** conformal https://arxiv.org/abs/2107.07511 · MAPIE https://github.com/scikit-learn-contrib/MAPIE · GPyTorch https://gpytorch.ai/ · BoTorch https://botorch.org/ · Murphy PML https://probml.github.io/pml-book/ · Pyro https://pyro.ai/ · **Applied:** AiZynthFinder https://github.com/MolecularAI/aizynthfinder · Open Reaction Database https://open-reaction-database.org/ · BELKA (DEL) https://www.kaggle.com/competitions/leash-BELKA · Flower (FL) https://flower.ai/ · **Benchmarks/datasets:** TDC https://tdcommons.ai/ · MoleculeNet https://moleculenet.org/ · OGB https://ogb.stanford.edu/ · Polaris https://polarishub.io/ · QM9 https://quantum-machine.org/datasets/ · PCQM4Mv2 https://ogb.stanford.edu/kddcup2021/pcqm4m/ · PDBbind https://www.pdbbind-plus.org.cn/ · DUD-E http://dude.docking.org/ · ChEMBL https://www.ebi.ac.uk/chembl/ · ZINC https://zinc.docking.org/ · **Protein (context):** AlphaFold 2 https://www.nature.com/articles/s41586-021-03819-2 · ESM https://github.com/facebookresearch/esm · ESM Atlas https://esmatlas.com/ · Boltz https://github.com/jwohlwend/boltz · OpenFold https://github.com/aqlaboratory/openfold

**Quantum & QML (Phase Q)** — IBM Quantum Learning https://learning.quantum.ibm.com/ · Watrous https://www.youtube.com/playlist?list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR · Basics https://learning.quantum.ibm.com/course/basics-of-quantum-information · Algorithms https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms · General Formulation https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information · Error Correction https://learning.quantum.ibm.com/course/foundations-of-quantum-error-correction · Qiskit https://www.youtube.com/@qiskit/playlists · QDA/SQD blog https://www.ibm.com/quantum/blog/iql-migration · PennyLane codebook https://pennylane.ai/codebook · PennyLane QML https://pennylane.ai/topics/quantum-machine-learning · PennyLane chemistry https://pennylane.ai/qml/demos_quantum-chemistry · MIT 8.04 https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2013/ · Quantum Country https://quantum.country/qcvc · Schuller https://www.youtube.com/playlist?list=PLPH7f_7ZlzxTi6kS4vCmv4ZKm9u8g5yic · Péré https://github.com/Christophe-pere/QML-Course · Hjorth-Jensen https://github.com/CompPhysics/QuantumComputingMachineLearning · VQE https://www.nature.com/articles/ncomms5213 · ADAPT-VQE https://www.nature.com/articles/s41467-019-10988-2 · DMET-VQE https://arxiv.org/abs/2108.08987 · QAOA https://arxiv.org/abs/1411.4028 · Hadfield https://arxiv.org/abs/1709.03489 · Pauli grouping https://arxiv.org/abs/1907.13117 · QML=kernels https://arxiv.org/abs/2101.11020 · Barren plateaus https://www.nature.com/articles/s41467-018-07090-4 · Expressibility https://arxiv.org/abs/1905.10876 · Advantage-goal https://arxiv.org/abs/2203.01340 · PQC encodings https://arxiv.org/abs/2008.08605 · Preskill Ph219 http://theory.caltech.edu/~preskill/ph219/ · TensorFlow Quantum https://www.tensorflow.org/quantum

**GenAI & Agents (P6)** — HF LLM course https://huggingface.co/learn/llm-course · HF Agents https://huggingface.co/learn/agents-course · HF MCP https://huggingface.co/learn/mcp-course · Anthropic Effective Agents https://www.anthropic.com/research/building-effective-agents · DeepLearning.AI https://www.deeplearning.ai/short-courses/ · LangGraph https://langchain-ai.github.io/langgraph/ · LoRA https://arxiv.org/abs/2106.09685 · QLoRA https://arxiv.org/abs/2305.14314 · RLHF https://arxiv.org/abs/2203.02155 · DPO https://arxiv.org/abs/2305.18290 · CLIP https://arxiv.org/abs/2103.00020 · LLaVA https://arxiv.org/abs/2304.08485 · Flamingo https://arxiv.org/abs/2204.14198 · PMC-OA https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ · PubMed E-utilities https://www.ncbi.nlm.nih.gov/books/NBK25501/ · Semantic Scholar API https://www.semanticscholar.org/product/api

**CV expert / 3D (P7)** — FPCV https://fpcv.cs.columbia.edu/ · Coursera https://www.coursera.org/specializations/firstprinciplesofcomputervision · Stachniss https://www.youtube.com/@CyrillStachniss · Michigan EECS 498 https://web.eecs.umich.edu/~justincj/teaching/eecs498/ · playlist https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r · Szeliski https://szeliski.org/Book/ · COLMAP https://colmap.github.io/ · OpenMVG https://github.com/openMVG/openMVG · ORB-SLAM3 https://github.com/UZ-SLAMLab/ORB_SLAM3 · Open3D http://www.open3d.org/ · Detectron2 https://github.com/facebookresearch/detectron2 · MMDetection https://github.com/open-mmlab/mmdetection · YOLO https://docs.ultralytics.com/ · OpenCV https://docs.opencv.org/ · PyImageSearch https://pyimagesearch.com/ · ONNX https://onnx.ai/ · TensorRT https://developer.nvidia.com/tensorrt · CUDA Guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/ · NVIDIA DLI https://www.nvidia.com/en-us/training/ · Triton https://github.com/triton-inference-server/server · DeepStream https://developer.nvidia.com/deepstream-sdk · NeRF https://arxiv.org/abs/2003.08934 · 3DGS https://arxiv.org/abs/2308.04079 · nerfstudio https://docs.nerf.studio/ · KITTI https://www.cvlibs.net/datasets/kitti/ · TUM RGB-D https://cvg.cit.tum.de/data/datasets/rgbd-dataset · ETH3D https://www.eth3d.net/

**DSA & C++ (Thread T)** — learncpp https://www.learncpp.com/ · The Cherno https://www.youtube.com/@TheCherno · cppreference https://en.cppreference.com/ · CppCon https://www.youtube.com/@CppCon · NeetCode https://neetcode.io/ · LeetCode https://leetcode.com/ · MIT 6.006 https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ · Abdul Bari https://www.youtube.com/@abdul_bari · DesignGurus https://www.designgurus.io/ · Striver https://takeuforward.org/ · Sean Prashad https://seanprashad.com/leetcode-patterns/ · CP Handbook https://cses.fi/book/book.pdf · CSES https://cses.fi/problemset/

**Implement-by-hand (Thread M)** — d2l.ai https://d2l.ai/ · labml.ai https://nn.labml.ai/ · Annotated Transformer http://nlp.seas.harvard.edu/annotated-transformer/ · micrograd https://github.com/karpathy/micrograd · nanoGPT https://github.com/karpathy/nanoGPT · ML-From-Scratch https://github.com/eriklindernoren/ML-From-Scratch · minGPT https://github.com/karpathy/minGPT · tinygrad https://github.com/tinygrad/tinygrad · JAX https://jax.readthedocs.io/ · Equinox https://docs.kidger.site/equinox/ · Anki https://apps.ankiweb.net/

**Research output (Thread R)** — Papers with Code https://paperswithcode.com/ · MICCAI https://www.miccai.org/ · MIDL https://midl.io/ · MLSB https://www.mlsb.io/ · LoG https://logconference.org/ · ISCB/ISMB https://www.iscb.org/ · arXiv https://arxiv.org/ · bioRxiv https://www.biorxiv.org/ · OpenReview https://openreview.net/ · How to Read a Paper https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf · Connected Papers https://www.connectedpapers.com/ · Papers We Love https://github.com/papers-we-love/papers-we-love

**Interview (P8)** — NeetCode https://neetcode.io/ · LeetCode https://leetcode.com/ · DesignGurus https://www.designgurus.io/ · Striver https://takeuforward.org/ · ByteByteGo https://bytebytego.com/ · https://www.youtube.com/@ByteByteGo · Gaurav Sen https://www.youtube.com/@gkcs · MIT 6.824 https://pdos.csail.mit.edu/6.824/

**Company ladder** — Qure.ai https://qure.ai/ · SigTuple https://sigtuple.com/ · NVIDIA https://www.nvidia.com/en-us/about-nvidia/careers/ · GE HealthCare https://www.gehealthcare.com/ · MSR India https://www.microsoft.com/en-us/research/lab/microsoft-research-india/ · Google Research India https://research.google/locations/india/ · Owkin https://www.owkin.com/ · Recursion https://www.recursion.com/careers · Insitro https://insitro.com/ · Genesis https://www.genesistherapeutics.ai/ · Iambic https://www.iambic.ai/ · Chai https://www.chaidiscovery.com/ · EvolutionaryScale https://www.evolutionaryscale.ai/ · Profluent https://www.profluent.bio/ · InstaDeep https://www.instadeep.com/ · Valence Labs https://www.valencelabs.com/ · Isomorphic Labs https://www.isomorphiclabs.com/careers · Google DeepMind https://deepmind.google/about/careers/ · QpiAI https://www.qpiai.tech/careers · IBM Quantum https://www.ibm.com/quantum · IISc ERP https://www.iisc.ac.in/admissions/external-registration-programme-ph-d/ · EURAXESS https://euraxess.ec.europa.eu/

---

# A closing note
This detailed edition trades the crispness of the summary v12 for the thing you actually study from: every phase now reads like Phase 5 — week-by-week, each resource carrying its real hour cost and a link, each week ending in something you *make* and something you must be able to *explain unaided*. The shape is deliberate. Hard theory goes in the fresh morning block; building goes in the second; review and reading go in the tired evening; the weekend is a buffer, not a debt. The self-tests, not the calendar, are the gates — if one doesn't pass, the block stretches. Protect sleep and the rest day like production infrastructure, because at 40 hours a week on top of a job they are exactly that. And keep every portfolio artifact public and yours to defend — that single discipline is what turns two years of work into a resume and a set of interviews that hold up under the hardest questioning. Adjust as the squads' priorities move; the public skills here are the durable core beneath whatever the project names happen to be.
