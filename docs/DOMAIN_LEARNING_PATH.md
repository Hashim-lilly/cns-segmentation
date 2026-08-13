# Domain Knowledge Learning Path — CNS Drug Delivery & Intrathecal CFD

> **Why this matters:** You can't build a robust segmentation pipeline without understanding what you're segmenting, why the geometry matters for drug delivery, and how errors propagate into CFD simulation outputs. This isn't optional background — it's the foundation that makes every technical decision defensible.

---

## Learning Tracks (Recommended Order)

### Track 1: Neuroanatomy of the Spine & CSF System (Week 1)
### Track 2: MRI Physics — Why Images Look the Way They Do (Week 1-2)
### Track 3: CSF Dynamics & Intrathecal Drug Delivery (Week 2)
### Track 4: CFD Fundamentals — What the Simulation Actually Does (Week 2-3)
### Track 5: Medical Image Segmentation & Deep Learning (Week 3-4)
### Track 6: The Business Case — Why Lilly Cares About This (Week 1, revisit)

---

## Track 1: Neuroanatomy — The Structures You're Segmenting

**Goal:** Know what the spinal cord, dura, CSF, nerve rootlets, and subarachnoid space look like and why they matter.

### 🎥 Video Resources (Free)

| # | Resource | Platform | Duration | What You'll Learn |
|---|----------|----------|----------|-------------------|
| 1 | **[Anatomy of the Spinal Cord — Cross Section](https://www.youtube.com/watch?v=V4ho5E3wXSk)** | YouTube (Ninja Nerd) | 25 min | Spinal cord structure, gray/white matter, anterior/posterior horns |
| 2 | **[Meninges of the Spinal Cord](https://www.youtube.com/watch?v=HT4MbX20r3E)** | YouTube (Ninja Nerd) | 20 min | **THE KEY VIDEO** — dura, arachnoid, pia mater, subarachnoid space, CSF location |
| 3 | **[Spinal Nerve Roots and Dermatomes](https://www.youtube.com/watch?v=yFQ0WLx_j3I)** | YouTube (Ninja Nerd) | 30 min | Nerve rootlets (dorsal/ventral), how they exit, why they matter for drug mixing |
| 4 | **[Cerebrospinal Fluid (CSF) — Production, Circulation, Absorption](https://www.youtube.com/watch?v=oeEaaWbhv3E)** | YouTube (Ninja Nerd) | 35 min | Where CSF is made (choroid plexus), how it flows, where it's absorbed |
| 5 | **[Ventricular System of the Brain](https://www.youtube.com/watch?v=GIJF7Tqry48)** | YouTube (Ninja Nerd) | 25 min | Lateral/3rd/4th ventricles, aqueduct — Phase 2 structures |
| 6 | **[Lumbar Puncture Anatomy](https://www.youtube.com/watch?v=rGaFvYbShZ0)** | YouTube (Armando Hasudungan) | 10 min | WHY drugs are injected at L3-L4 (below conus medullaris, into the cauda equina) |
| 7 | **[Stanford Human Anatomy — Vertebral Column](https://www.youtube.com/watch?v=k0T3dj66Pzo)** | YouTube (Stanford Medicine) | 50 min | Full vertebral column lecture from Stanford |

### 📖 Quick Reads
- **[Radiopaedia: Subarachnoid Space](https://radiopaedia.org/articles/subarachnoid-space)** — 5-min read, annotated images
- **[Radiopaedia: CSF Flow](https://radiopaedia.org/articles/cerebrospinal-fluid-1)** — Production, circulation, absorption
- **[TeachMeAnatomy: Spinal Meninges](https://teachmeanatomy.info/neuro/structures/meninges/)** — Clear diagrams of dura/arachnoid/pia

### 🎯 After Track 1, you should be able to:
- [ ] Draw a cross-section of the spinal canal labeling: dura, arachnoid, pia, CSF space, cord, rootlets
- [ ] Explain why the subarachnoid space IS the fluid domain for CFD
- [ ] Explain why nerve rootlets increase drug mixing by 60%
- [ ] Explain why lumbar puncture happens at L3-L4

---

## Track 2: MRI Physics — Why Images Look the Way They Do

**Goal:** Understand T1 vs T2 contrast, why CSF is bright on T2, what voxel spacing means, and how different sequences affect segmentation.

### 🎥 Video Resources (Free)

| # | Resource | Platform | Duration | What You'll Learn |
|---|----------|----------|----------|-------------------|
| 1 | **[MRI Physics — How Does MRI Work?](https://www.youtube.com/watch?v=djAxjtN_7VE)** | YouTube (Radiology Tutor) | 15 min | Basic principle (protons, magnetic field, RF pulses) — just enough |
| 2 | **[T1 vs T2 Weighted Images — EXPLAINED](https://www.youtube.com/watch?v=aCj1f6-VEEk)** | YouTube (Radiology Tutor) | 12 min | **CRITICAL** — why CSF is bright on T2 and dark on T1 |
| 3 | **[MRI Made Easy — Complete Course](https://www.youtube.com/playlist?list=PLPcImQzEnTpz-5TzxWYv8d3a9buyYiXBG)** | YouTube (Radiology Channel) | 2 hrs (playlist) | Full beginner MRI course, watch first 4-5 videos |
| 4 | **[MIT 2.71 — Intro to Medical Imaging (Lecture 18: MRI)](https://www.youtube.com/watch?v=fMWk5p0bGkk)** | MIT OCW | 50 min | University-level MRI physics |
| 5 | **[Spinal Cord MRI — Normal Anatomy](https://www.youtube.com/watch?v=LYG5W_hfJBg)** | YouTube (Radiology Assistant) | 10 min | How the spine looks on MRI specifically |

### 📖 Quick Reads
- **[Questions and Answers in MRI — mriquestions.com](http://mriquestions.com/index.html)** — The best free MRI physics reference
- **[NiBabel Coordinate Systems](https://nipy.org/nibabel/coordinate_systems.html)** — What the affine matrix means

### 🎓 Free Courses
| Course | Platform | Duration | Notes |
|--------|----------|----------|-------|
| **[Fundamentals of Biomedical Imaging: MRI](https://www.edx.org/learn/biomedical-sciences/ecole-polytechnique-federale-de-lausanne-fundamentals-of-biomedical-imaging-magnetic-resonance-imaging-mri)** | edX (EPFL) | 5 weeks | Free to audit; excellent Swiss engineering perspective |
| **[Introduction to Medical Imaging](https://www.coursera.org/learn/introduction-to-medical-imaging)** | Coursera (Stanford) | ~10 hrs | Free audit; covers X-ray, CT, MRI, ultrasound |

### 🎯 After Track 2, you should be able to:
- [ ] Explain why T2-weighted MRI is used for CSF segmentation (CSF = bright)
- [ ] Explain what voxel spacing (0.5×0.5×3mm) means and why anisotropic resolution matters
- [ ] Explain why different scanner vendors produce different-looking images
- [ ] Read a NIfTI header and interpret shape, spacing, affine orientation

---

## Track 3: CSF Dynamics & Intrathecal Drug Delivery

**Goal:** Understand how CSF flows, why it matters for drug delivery, and what the business problem actually is.

### 🎥 Video Resources (Free)

| # | Resource | Platform | Duration | What You'll Learn |
|---|----------|----------|----------|-------------------|
| 1 | **[CSF Flow and Dynamics — Animated](https://www.youtube.com/watch?v=02BL-PB2mMA)** | YouTube (Osmosis) | 10 min | Animated CSF flow, oscillatory nature, cardiac-driven pulsation |
| 2 | **[Intrathecal Drug Delivery — Mechanism](https://www.youtube.com/watch?v=Cm1N3IXqYrE)** | YouTube (Medical Animations) | 5 min | How drugs are injected into the CSF space and why |
| 3 | **[Antisense Oligonucleotides (ASOs) — Mechanism of Action](https://www.youtube.com/watch?v=5V-fG8mVRaw)** | YouTube (Ionis Pharmaceuticals) | 5 min | What ASO drugs are (Lilly has IT-delivered ASOs in pipeline) |
| 4 | **[Nusinersen (Spinraza) — The First Intrathecal ASO](https://www.youtube.com/watch?v=NfUMslNi_7w)** | YouTube (Biogen) | 3 min | Real-world example of IT drug that uses the same physics |
| 5 | **[Quigley et al. 2014 — CSF Biomechanics for Neuroradiologists](https://www.ajnr.org/content/35/10/1864)** | AJNR (free PDF) | 30 min read | THE BEST single paper for understanding CSF mechanics from a clinical perspective |

### 📖 Papers (All Open Access — Read These)
1. **[Sass et al. 2017](https://doi.org/10.1186/s12987-017-0085-y)** — The anchor paper. 3D SAS model with rootlets. Read the Introduction and Discussion.
2. **[Khani et al. 2022](https://doi.org/10.1186/s12987-022-00304-4)** — Human in-silico trials comparing LP vs CM vs ICV injection. Read for clinical relevance.
3. **[Pardridge 2005](https://doi.org/10.1602/neurorx.2.1.3)** — "98% of small molecules and 100% of large molecules cannot cross the BBB" — this is WHY intrathecal delivery exists.

### 🎯 Key Concepts to Internalize:
- **Why intrathecal?** The blood-brain barrier blocks >98% of drugs. Direct CSF injection bypasses it.
- **Why does flow matter?** CSF doesn't flow like blood (it *oscillates*). Drugs injected at L3-L4 must travel UP to the brain via *steady streaming* — a slow net drift caused by nonlinear oscillatory effects.
- **Why rootlets?** They obstruct the channel and create mixing (like rocks in a stream). Khani showed rootlets increase drug spread by 60%.
- **Why patient-specific?** Anatomy varies (CSF volume, cord eccentricity, compliance). Different patients get different drug exposure from the same injection.

---

## Track 4: CFD Fundamentals — What the Simulation Does

**Goal:** Understand what CFD solves, why mesh quality matters, and what OpenFOAM produces.

### 🎥 Video Resources (Free)

| # | Resource | Platform | Duration | What You'll Learn |
|---|----------|----------|----------|-------------------|
| 1 | **[What is CFD? (Computational Fluid Dynamics)](https://www.youtube.com/watch?v=YVLB3FJbWfE)** | YouTube (SimScale) | 8 min | Clear intro to what CFD is and isn't |
| 2 | **[Navier-Stokes Equations — Intuitive Explanation](https://www.youtube.com/watch?v=ERBVFcutl3M)** | YouTube (3Blue1Brown) | 25 min | Beautiful visual explanation of the math |
| 3 | **[CFD Meshing Explained](https://www.youtube.com/watch?v=5x1rnT0BmXM)** | YouTube (SimScale) | 12 min | Why mesh quality determines simulation quality |
| 4 | **[OpenFOAM Tutorial for Beginners](https://www.youtube.com/watch?v=BCXK7ynF5tE)** | YouTube (Jozsef Nagy) | 30 min | What an OpenFOAM case looks like, basic setup |
| 5 | **[Reynolds Number Explained](https://www.youtube.com/watch?v=y0WRJtXvpSo)** | YouTube (Real Engineering) | 10 min | What Re means and why CSF flow is laminar |
| 6 | **[MIT 2.29 — Numerical Fluid Mechanics (Lecture 1)](https://ocw.mit.edu/courses/2-29-numerical-fluid-mechanics-spring-2015/video_galleries/lecture-videos/)** | MIT OCW | 1.5 hrs | Full university CFD course (watch Lectures 1-3 for foundations) |

### 🎓 Free Courses
| Course | Platform | Duration | Notes |
|--------|----------|----------|-------|
| **[A Hands-on Introduction to CFD using OpenFOAM](https://www.youtube.com/playlist?list=PLvUvVMohnIMhsZwdIFsfKaBW9DP4mWo9y)** | YouTube (OpenFOAM Workshop) | 10+ hrs | Full OpenFOAM tutorial series |
| **[12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)** | Lorena Barba (GWU) | Self-paced | Python-based CFD fundamentals — excellent for building intuition |

### 🎯 After Track 4, you should be able to:
- [ ] Explain what Navier-Stokes equations predict (velocity and pressure everywhere in the fluid)
- [ ] Explain why a hole in the mesh causes the simulation to crash (fluid "leaks")
- [ ] Explain what Reynolds number means for CSF (Re<200 → laminar → predictable)
- [ ] Explain why boundary layers matter at the cord and dura walls (wall shear stress)
- [ ] Explain why staircase artifacts from voxel boundaries create spurious turbulence

---

## Track 5: Medical Image Segmentation & Deep Learning

**Goal:** Understand U-Net, nnU-Net, loss functions, and why topology matters for our use case.

### 🎥 Video Resources (Free)

| # | Resource | Platform | Duration | What You'll Learn |
|---|----------|----------|----------|-------------------|
| 1 | **[Deep Learning for Medical Image Segmentation — Full Course](https://www.youtube.com/watch?v=BNHR_GGLQHQ)** | YouTube (MONAI Bootcamp) | 2 hrs | THE best free resource — from the MONAI team directly |
| 2 | **[U-Net Paper Explained](https://www.youtube.com/watch?v=oLvmLJkmXuc)** | YouTube (Yannic Kilcher) | 30 min | Original U-Net architecture that everything builds on |
| 3 | **[nnU-Net Explained — Self-Configuring Segmentation](https://www.youtube.com/watch?v=BN_Tvc_tSeo)** | YouTube (AI Coffee Break) | 15 min | Why nnU-Net wins every benchmark and how it self-configures |
| 4 | **[Dice Loss vs Cross-Entropy — When to Use What](https://www.youtube.com/watch?v=BT21g_wWjHo)** | YouTube (AIFinder) | 12 min | Loss functions for segmentation |
| 5 | **[3D Medical Image Segmentation with MONAI](https://www.youtube.com/watch?v=M3ZWfamWrBM)** | YouTube (Project MONAI) | 45 min | How to actually train a 3D segmentation model |
| 6 | **[Stanford CS231n — Segmentation (Lecture 11)](https://www.youtube.com/watch?v=nDPWywWRIRo)** | YouTube (Stanford) | 75 min | University-level semantic segmentation lecture |

### 🎓 Free Courses
| Course | Platform | Duration | Notes |
|--------|----------|----------|-------|
| **[AI for Medicine Specialization](https://www.coursera.org/specializations/ai-for-medicine)** | Coursera (deeplearning.ai) | 3 courses | Free to audit; Course 1 covers medical image segmentation |
| **[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)** | Coursera (Andrew Ng) | 5 courses | Free to audit; Courses 1-4 for deep learning foundations |
| **[MONAI Tutorials (Runnable Notebooks)](https://github.com/Project-MONAI/tutorials)** | GitHub | Self-paced | Start with `3d_segmentation/spleen_segmentation_3d.ipynb` |

### 🎯 After Track 5, you should be able to:
- [ ] Draw a U-Net architecture and explain skip connections
- [ ] Explain why Dice loss handles class imbalance (tiny cord in large volume)
- [ ] Explain why clDice loss preserves thin structures (rootlets)
- [ ] Explain what "watertight mesh" means and why CFD needs it

---

## Track 6: The Business Case — Why Lilly Invests in This

**Goal:** Understand the pharmaceutical context — what decision this pipeline informs.

### 🎥 Video Resources (Free)

| # | Resource | Platform | Duration | What You'll Learn |
|---|----------|----------|----------|-------------------|
| 1 | **[The Blood-Brain Barrier Problem in Drug Delivery](https://www.youtube.com/watch?v=Bx4bM6jJ2Xw)** | YouTube (Nature Video) | 5 min | Why CNS drug delivery is the hardest problem in pharma |
| 2 | **[In Silico Clinical Trials — FDA Perspective](https://www.youtube.com/watch?v=FG-YWX5kp_c)** | YouTube (FDA) | 15 min | How computational models inform regulatory decisions |
| 3 | **[Digital Twins in Healthcare](https://www.youtube.com/watch?v=2l_K2B7vY3A)** | YouTube (Siemens Healthineers) | 10 min | The broader vision: patient-specific computational models |
| 4 | **[ASME V&V 40 — Computational Modeling Credibility](https://www.youtube.com/watch?v=6l_g5bHi-yI)** | YouTube (ASME) | 45 min | The framework pharma uses to validate computational models |

### 📖 Read Your Own Papers
- **Montoya, Teli et al. (2024)** — Lilly's own CSF model paper (Paper 1 in your backup)
- **Khani et al. (2025)** — NHP digital twin (Paper 2) — shows where the field is heading

### 🎯 The business logic in one paragraph:
> Lilly develops drugs that must be injected into the CSF (intrathecally) because they can't cross the blood-brain barrier. The question is: *where in the brain/spine does the drug actually go, how fast, and at what concentration?* The answer depends on patient-specific anatomy (geometry) + CSF flow (physics). A CFD simulation predicts this — but needs an accurate 3D model of the patient's spinal CSF space. Building that model from MRI is what our segmentation pipeline does. Better geometry → better simulation → better dosing decisions → fewer failed trials.

---

## Recommended Learning Schedule (2 Weeks Intensive)

### Week 1: Anatomy + MRI + Business Context

| Day | Morning (2-3 hrs) | Afternoon (1-2 hrs) |
|-----|-------------------|---------------------|
| **Mon** | Track 1: Videos 1-3 (spinal cord, meninges, rootlets) | Track 6: BBB video + read Montoya paper intro |
| **Tue** | Track 1: Videos 4-6 (CSF, ventricles, lumbar puncture) | Track 2: Videos 1-2 (MRI basics, T1 vs T2) |
| **Wed** | Track 2: Videos 3-4 (MRI course playlist) | Track 3: Videos 1-2 (CSF flow, IT delivery) |
| **Thu** | Track 3: Read Quigley 2014 paper + Sass 2017 intro | Track 3: Videos 3-5 (ASOs, Spinraza) |
| **Fri** | Track 2: EPFL edX course (Module 1) | **Self-test:** Can you explain what the pipeline does and why? |

### Week 2: CFD + Segmentation + Hands-On

| Day | Morning (2-3 hrs) | Afternoon (2-3 hrs) |
|-----|-------------------|---------------------|
| **Mon** | Track 4: Videos 1-3 (CFD intro, Navier-Stokes, meshing) | Track 4: Lorena Barba Steps 1-4 (CFD Python) |
| **Tue** | Track 4: Videos 4-5 (OpenFOAM, Reynolds number) | Track 5: Video 1 (MONAI Bootcamp — medical seg) |
| **Wed** | Track 5: Videos 2-3 (U-Net, nnU-Net) | Track 5: MONAI spleen tutorial (run it yourself) |
| **Thu** | Track 5: Videos 4-6 (loss functions, 3D seg, Stanford) | **Hands-on:** Run notebook 01 + 05 in this project |
| **Fri** | Track 4: MIT OCW Lecture 1 (deeper CFD) | **Integration:** Read Sass 2017 fully — connect all tracks |

---

## The "Aha Moment" Connections

Once you've gone through the tracks, these connections should click:

1. **Why T2w MRI?** → Because CSF is bright (long T2 relaxation of water) and cord is dark → easy boundary detection

2. **Why Boolean subtraction (canal − cord)?** → The CSF *is* the space between the cord (inner wall) and dura (outer wall). Subtracting gives the fluid domain.

3. **Why rootlets matter (+60% drug spread)?** → They're obstacles in the annular channel. Like rocks in a river, they create vortices and mixing that accelerates transport.

4. **Why watertight mesh?** → The Navier-Stokes solver computes pressure and velocity *everywhere inside* the mesh. A hole means fluid escapes → infinite velocity → simulation crashes.

5. **Why Womersley number > 5?** → CSF flow is *inertia-dominated oscillatory* (not steady Poiseuille). This means the velocity profile is flat in the center, not parabolic. This matters for how drug gets carried.

6. **Why patient-specific?** → CSF volume varies 2-3× across patients. A narrow canal means faster flow and different drug distribution. One-size-fits-all simulations mislead.

7. **Why automation over manual?** → Sass/Khani still use ITK-SNAP manually (days per case). We want minutes per case → enables population studies and clinical deployment.

---

## Bonus: Domain Expert YouTube Channels

Subscribe to these for ongoing learning:
- **[Ninja Nerd](https://www.youtube.com/c/NinjaNerdScience)** — Best anatomy/physiology channel
- **[3Blue1Brown](https://www.youtube.com/c/3blue1brown)** — Math intuition (Navier-Stokes, linear algebra)
- **[SimScale](https://www.youtube.com/c/SimScale)** — CFD tutorials and concepts
- **[Project MONAI](https://www.youtube.com/c/ProjectMONAI)** — Medical image AI (official)
- **[Radiology Tutor](https://www.youtube.com/c/RadiologyTutor)** — MRI physics clearly explained
- **[AI Coffee Break with Letitia](https://www.youtube.com/c/AICoffeeBreak)** — Paper explanations

---

*This learning path is designed to be executed in parallel with development — not as a prerequisite. Start building from Day 1; learn what you need just-in-time. But the anatomy and CSF dynamics tracks (1 + 3) should come FIRST because they answer "what am I looking at?" and "why does this matter?"*
