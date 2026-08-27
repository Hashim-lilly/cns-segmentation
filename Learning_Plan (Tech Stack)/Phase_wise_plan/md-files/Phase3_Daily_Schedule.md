# Phase 3 — Day-by-Day Schedule · Modern DL Theory + Foundation Architectures
### Weeks 31–36 · Mon Apr 5 → Sun May 16, 2027 · ~165 hrs · → Tier 1 (broad)

**Goal:** stop treating transformers and diffusion as black boxes — derive every layer of attention, understand optimization dynamics, and defend architecture choices on theory. Feeds Phase 4 and the generative parts of the molecular track (5C).

**Blocks:** **A** 06:00–08:00 (theory) · **B** 08:30–10:30 (build + threads) · **Evening** 20:30–22:30 (papers, Anki, R). **Weekend** = buffer + rest.
**Threads:** **T** DSA+C++ (Block B ~45 min) · **M** implement-from-scratch (Block B) · **R** research (Evening ~2 h/wk).

---

### Week 31 · Apr 5–11 — Optimization dynamics
| Day | Block A | Block B (T/M + build) | Evening (+ R) |
|---|---|---|---|
| Mon | Ruder: SGD → momentum | **M:** implement SGD + momentum from scratch | Anki |
| Tue | Adam family (RMSProp, Adam, AdamW) | **M:** implement Adam; compare on a 2D loss | Anki |
| Wed | Boyd: convex sets & functions | **T:** DP → visualize convex vs non-convex | Boyd lecture; Anki |
| Thu | Gradient/Newton; conditioning | **M:** LR-warmup + schedule experiment | Anki |
| Fri | Why DL trains despite non-convexity | **T:** timed set | Whiteboard-Fri: why momentum accelerates; what Adam's 2nd moment does |
| Sat–Sun | **Buffer + rest** | | |

### Week 32 · Apr 12–18 — Information theory
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | MacKay ch 1–2: entropy, information | **T:** graphs → implement entropy/CE/KL cleanly | Anki |
| Tue | KL divergence; cross-entropy from KL | **M:** KL between two Gaussians | Anki |
| Wed | Mutual information | **T:** DP → MI estimate on a toy set | Anki |
| Thu | MI in contrastive losses (InfoNCE) | **M:** a tiny InfoNCE loss | **R:** reproduce-paper |
| Fri | Coding & compression view | **T:** timed set | Whiteboard-Fri: derive cross-entropy from KL; NLL = MLE |
| Sat–Sun | **Buffer + rest** | | |

### Week 33 · Apr 19–25 — Transformers in depth
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | CS25: attention & the transformer | **M:** multi-head attention as one matmul | Annotated Transformer |
| Tue | Positional encodings; RoPE | **M:** sinusoidal + RoPE from scratch | Anki |
| Wed | RMSNorm; pre-LN vs post-LN | **T:** intervals → add RMSNorm to your block | Lilian Weng blog |
| Thu | FlashAttention; GQA/MQA (concepts) | **M:** SwiGLU FFN; assemble a full block | Anki; **R:** write-up |
| Fri | MoE / Switch (concept) | **T:** timed set | Whiteboard-Fri: derive attention; param-count a block |
| Sat–Sun | **Buffer + rest** · *Deliverable:* a from-scratch transformer block (attention+RoPE+RMSNorm+SwiGLU) | | |

### Week 34 · Apr 26–May 2 — Vision transformers + modern ConvNets
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | ViT lectures (patch embedding) | **M:** ViT patch-embed + classifier head | Anki |
| Tue | ConvNeXt (a modernized CNN) | **T:** graphs → run a ConvNeXt baseline | Anki |
| Wed | DINOv2 (SSL) | **M:** DINOv2 features + linear probe | Anki |
| Thu | MAE (masked autoencoding) | **T:** DP → an MAE-style masking demo | **R:** reproduce-paper |
| Fri | CNN vs ViT inductive biases | **T:** timed set | Whiteboard-Fri: why SSL helps in low-data (ties to imaging FMs) |
| Sat–Sun | **Buffer + rest** | | |

### Weeks 35–36 · May 3–16 — Diffusion & generative modeling (feeds 5C)
| Day | Block A | Block B | Evening (+ R) |
|---|---|---|---|
| Mon (Wk 35) | CS236: latent-variable models, VAEs | **M:** a small VAE | Anki |
| Tue | Normalizing flows (concept) | **T:** DP → flow toy | Anki |
| Wed | DDPM: forward/reverse process | **M:** DDPM forward (noising) | DDPM paper |
| Thu | DDPM: the ε-prediction loss | **M:** DDPM reverse + ε-loss | Anki |
| Fri | Score-based view (Yang Song) | **T:** timed set | Whiteboard-Fri: derive the diffusion loss (both views) |
| Mon–Thu (Wk 36) | Guidance; DDIM sampling; conditioning | **M:** train the DDPM on a toy set; sampling; **T:** 1 set/day | HF Diffusion course; **R:** write-up |
| Fri (Wk 36) | **🚩 PHASE-3 SELF-TEST** (attention on whiteboard in 5 min; derive diffusion loss; LayerNorm-not-BatchNorm in transformers; positional encodings compared; param-count a block) | Polish a minimal-DDPM repo | **R:** post write-up |
| Sat–Sun | **Buffer + rest** · *Deliverable:* minimal DDPM (from scratch) trained on a toy set | | |

**End of Phase 3 → Phase 4 (Production AI Systems + MLOps) next.**
