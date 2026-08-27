# Phase 5 — Day-by-Day Schedule · Molecular-ML / Drug-Discovery Track (5A–5F)
### Weeks 43–57 · Mon Jun 28 → Sun Oct 10, 2027 · ~450 hrs · → Tier 2 (the DESRES / Isomorphic differentiator)

**Goal:** across public data — property prediction (fingerprint→D-MPNN→3D-equivariant), molecular foundation models, generative chemistry (GFlowNets/diffusion), the physics stack (conformer→docking→FEP→MD), calibrated UQ, and applied cheminformatics. Every week ends in something you build and can defend cold.

*Blocks: **A** 06–08 (theory) · **B** 08:30–10:30 (build + threads) · **Evening** 20:30–22:30 (papers, Anki, R). Weekend = buffer + rest (this phase is dense — use it). Threads: **T** DSA (Block B ~45m) · **M** implement (Block B) · **R** research/apply (Evening).*

---

## 5A — Geometric & molecular ML core (Weeks 43–45)

### Week 43 · Jun 28–Jul 4 — Graph ML foundations + molecular representations
| Day | Block A | Block B (T/M) | Evening (+ R) |
|---|---|---|---|
| Mon | CS224W: node/graph embeddings | **T:** graphs → PyG GCN on a toy set | Anki |
| Tue | Message passing; GNN layers | **M:** a message-passing layer from scratch | Anki |
| Wed | GNN expressiveness (Weisfeiler-Lehman) | **T:** DP → GraphSAGE/GAT | Anki |
| Thu | Molecular representations: fingerprints/descriptors | **M:** RDKit Morgan + descriptors featurizer | **R:** pick MLSB-track paper |
| Fri | Scaffold splits (the honest split) | **T:** timed set | Whiteboard-Fri: message-passing update + permutation invariance |
| Sat–Sun | **Buffer + rest** · *Deliverable:* baselines (fingerprint+GBM) on a TDC ADMET task | | |

### Week 44 · Jul 5–11 — Message-passing for molecules (D-MPNN / Chemprop)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Chemprop / D-MPNN paper | **M:** train Chemprop on your TDC task | Anki |
| Tue | Directed messages; tottering | **T:** graphs → ablate directed vs undirected | Anki |
| Wed | Feature concatenation; readouts | **M:** finish from-scratch MPNN; beat baseline | Anki |
| Thu | Graph transformers (Graphormer/GPS) | **T:** DP → run a graph-transformer demo | **R:** reproduce-paper |
| Fri | Over-smoothing + fixes | **T:** timed set | Whiteboard-Fri: derive D-MPNN edge update |
| Sat–Sun | **Buffer + rest** | | |

### Week 45 · Jul 12–18 — 3D / equivariant nets + 🎯 5A capstone
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Geometric DL (Bronstein): equivariance | **M:** verify equivariance empirically | Anki |
| Tue | SchNet → DimeNet++ | **T:** graphs → run SchNet on QM9 | Anki |
| Wed | EGNN (E(n)-equivariance) | **M:** an EGNN layer; QM9 property | Anki |
| Thu | NequIP / MACE (potentials) | **T:** DP → data-efficiency ablation | **R:** write-up |
| Fri | 2D vs 3D — when each wins | Build the **representation ladder** (fp vs D-MPNN vs 3D) | Whiteboard-Fri: invariance vs equivariance |
| Sat–Sun | **Buffer + rest** · *Deliverable:* 5A capstone (representation ladder, public repo) | | |

## 5B — Molecular foundation models & SSL (Weeks 46–47)

### Week 46 · Jul 19–25 — SSL objectives + SMILES/graph pretraining
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | ChemBERTa (masked-SMILES) | **M:** tokenize + mask a SMILES set | Anki |
| Tue | MolCLR (contrastive) | **T:** graphs → a contrastive augmentation | Anki |
| Wed | Grover / Uni-Mol (3D-aware) | **M:** small masked-SMILES pretrain run | Anki |
| Thu | Valid molecular augmentations | **T:** DP → augmentation checks | **R:** reproduce-paper |
| Fri | What each SSL signal captures | **T:** timed set | Whiteboard-Fri: the MLM objective |
| Sat–Sun | **Buffer + rest** | | |

### Week 47 · Jul 26–Aug 1 — Finetune + transfer study (🎯 5B capstone)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Fine-tuning strategies | **M:** finetune a pretrained encoder on a TDC task | Anki |
| Tue | Linear-probe vs full finetune | **T:** graphs → linear-probe baseline | Anki |
| Wed | Transfer at 100/1k/10k labels | **M:** data-ablation study | Anki |
| Thu | When SSL helps vs hurts | **T:** DP → compare vs from-scratch (5A) | **R:** write-up |
| Fri | Wrap 5B capstone | Polish repo + transfer-gain plot | Whiteboard-Fri: when a fingerprint+GBM still wins |
| Sat–Sun | **Buffer + rest** · *Deliverable:* 5B capstone (pretrain→finetune transfer gain) | | |

## 5C — Generative chemistry (Weeks 48–49)

### Week 48 · Aug 2–8 — Generative baselines + landscape + metrics
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | VAE/flow baselines (JT-VAE, MoFlow) | **M:** a small molecular VAE | Anki |
| Tue | Generative landscape for molecules | **T:** graphs → validity/novelty metrics | Anki |
| Wed | Metrics: validity/unique/novel/diverse | **M:** a metrics + reward module (QED/SA) | Anki |
| Thu | Reward design (property proxies) | **T:** DP → reward function | **R:** reproduce-paper |
| Fri | MOSES/GuacaMol benchmarks | **T:** timed set | Whiteboard-Fri: why high-reward alone → mode collapse |
| Sat–Sun | **Buffer + rest** | | |

### Week 49 · Aug 9–15 — GFlowNets + equivariant diffusion (🎯 5C capstone)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | GFlowNet theory (flow-matching / TB) | **M:** GFlowNet tutorial env | Anki |
| Tue | Reward-∝-diversity guarantee | **T:** graphs → GFlowNet on a fragment env | Anki |
| Wed | Equivariant diffusion (EDM/GeoDiff) | **M:** run a small 3D generation | Anki |
| Thu | Diffusion vs GFlowNet vs RL | **T:** DP → evaluate diversity/validity | **R:** write-up |
| Fri | Wrap 5C capstone | Polish generator + evaluation | Whiteboard-Fri: the GFlowNet objective (reward-proportional diversity) |
| Sat–Sun | **Buffer + rest** · *Deliverable:* 5C capstone (property-reward generator, public repo) | | |

## 5D — Physics-based structure & simulation (Weeks 50–53)

### Week 50 · Aug 16–22 — Conformer generation & 3D analysis
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | RDKit ETKDGv3; MMFF94s | **M:** generate + optimize conformers | Anki |
| Tue | Butina RMSD clustering | **T:** DP → cluster ensembles | Anki |
| Wed | COV/MAT vs GEOM-Drugs | **M:** COV/MAT/RMSD benchmark | Anki |
| Thu | DFT re-ranking (Psi4/PySCF) — your strength | **T:** run DFT single points on top-k | **R:** reproduce-paper |
| Fri | MMFF-vs-DFT ranking correlation | **T:** timed set | Whiteboard-Fri: why MMFF ordering is unreliable; COV vs MAT |
| Sat–Sun | **Buffer + rest** · *(follows the conformer capstone spec)* | | |

### Week 51 · Aug 23–29 — Molecular docking
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | AutoDock Vina: search + scoring | **M:** dock a public target/ligand set | Anki |
| Tue | Enrichment (EF/BEDROC), pose RMSD | **T:** DP → compute enrichment | Anki |
| Wed | gnina (CNN scoring) | **M:** re-score with gnina; compare | Anki |
| Thu | DiffDock (generative docking) | **T:** graphs → run DiffDock | **R:** write-up |
| Fri | What a docking score really approximates | **T:** timed set | Whiteboard-Fri: pose RMSD vs enrichment |
| Sat–Sun | **Buffer + rest** | | |

### Week 52 · Aug 30–Sep 5 — Free energy (FEP / ABFEP)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Stat-mech: free energy, ensembles | **M:** set up a toy alchemical transform | Anki |
| Tue | Thermodynamic cycles; error cancellation | **T:** DP → cycle bookkeeping | Anki |
| Wed | OpenFE: relative FEP setup | **M:** a small relative-FEP calc | Anki |
| Thu | Reading FEP results critically | **T:** graphs → analyze output | **R:** reproduce-paper |
| Fri | When FEP is worth the cost | **T:** timed set | Whiteboard-Fri: the thermodynamic cycle behind relative FEP |
| Sat–Sun | **Buffer + rest** | | |

### Week 53 · Sep 6–12 — Molecular dynamics + 🎯 5D capstone
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | OpenMM: force fields, integrators | **M:** a short protein–ligand MD | Anki |
| Tue | Thermostats/barostats; PBC | **T:** DP → set up an NVT/NPT run | Anki |
| Wed | MDAnalysis (RMSD/RMSF/contacts) | **M:** trajectory analysis | Anki |
| Thu | ML potentials (MACE-OFF/ANI) as surrogates | **T:** re-score with an ML potential | **R:** write-up |
| Fri | The full accuracy/cost ladder | Wrap 5D capstone (conformer→dock→MD + FEP) | Whiteboard-Fri: the ladder + where each breaks (ties to your VQE work) |
| Sat–Sun | **Buffer + rest** · *Deliverable:* 5D capstone (public target; pose/enrichment + MD + FEP) | | |

## 5E — Uncertainty quantification & probabilistic ML (Weeks 54–55)

### Week 54 · Sep 13–19 — Conformal prediction + calibration
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Conformal prediction (Angelopoulos) | **M:** split conformal on your 5A model | Anki |
| Tue | Coverage guarantees; exchangeability | **T:** DP → empirical coverage check | Anki |
| Wed | MAPIE hands-on | **M:** MAPIE intervals + plot | Anki |
| Thu | Calibration: ECE, reliability, temp-scaling | **T:** graphs → reliability diagram | **R:** reproduce-paper |
| Fri | Scaffold/temporal shift breaks it | **T:** timed set | Whiteboard-Fri: why split conformal gives valid coverage |
| Sat–Sun | **Buffer + rest** | | |

### Week 55 · Sep 20–26 — GP / SVGP + UQ comparison (🎯 5E capstone)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | GP regression: kernels, posterior | **M:** a GP on a small set (GPyTorch) | Anki |
| Tue | SVGP (sparse variational) for scale | **T:** DP → SVGP on ADMET | Anki |
| Wed | Deep ensembles vs MC-dropout | **M:** ensemble + MC-dropout UQ | Anki |
| Thu | Four-way UQ comparison | **T:** graphs → conformal/GP/dropout/ensemble | **R:** write-up |
| Fri | Which to ship when | Wrap 5E capstone | Whiteboard-Fri: GP posterior + what the kernel encodes |
| Sat–Sun | **Buffer + rest** · *Deliverable:* 5E capstone (conformal + SVGP on 5A model; coverage) | | |

## 5F — Applied cheminformatics (Weeks 56–57)

### Week 56 · Sep 27–Oct 3 — Retrosynthesis + reaction-condition + DEL
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | AiZynthFinder (MCTS retrosynthesis) | **M:** run retrosynthesis on public targets | Anki |
| Tue | Reaction representation + conditions | **T:** DP → reaction-condition model (USPTO/ORD) | Anki |
| Wed | DEL modeling; noisy counts | **M:** DEL enrichment on BELKA | Anki |
| Thu | Multitask / MMoE; contrastive | **T:** graphs → multi-endpoint ADMET | **R:** reproduce-paper |
| Fri | Synthesizability as a constraint | **T:** timed set | Whiteboard-Fri: MCTS retrosynthesis; DEL denoising |
| Sat–Sun | **Buffer + rest** | | |

### Week 57 · Oct 4–10 — Federated + virtual screening + 🎯 5F capstone + track self-test
| Day | Block A | Block B | Evening (+ R) |
|---|---|---|---|
| Mon | Federated learning (Flower) | **M:** a federated multi-endpoint ADMET demo | Anki |
| Tue | Active-learning virtual screening | **T:** DP → AL loop (dock + surrogate) | Anki |
| Wed | Wrap the 5F capstone (pick a lane) | **M:** MCP-wrapped property+UQ tool (or chosen lane) | Anki |
| Thu | Consolidate the whole track | Polish repos + write-ups | **R:** post; submit one capstone to MLSB/LoG |
| Fri | **🚩 PHASE-5 TRACK SELF-TEST** (message-passing vs conv; GFlowNet objective; the accuracy/cost ladder; conformal coverage; SVGP vs dropout; MCTS/DEL/MMoE/federated) | Final polish | **▶ Tier-2 applications** (DESRES, Isomorphic, DeepMind-science) |
| Sat–Sun | **Buffer + rest** · *Deliverable:* 5F capstone (public) | | |

**End of Phase 5 → Phase 6 (Scientific / chemistry agents) next. Tier 2 reached — drug-discovery credible.**
