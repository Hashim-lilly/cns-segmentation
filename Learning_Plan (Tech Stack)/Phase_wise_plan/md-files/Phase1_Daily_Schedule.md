# Phase 1 — Day-by-Day Schedule · Mathematics + NNs from First Principles
### Weeks 3–12 · Mon Sep 21 → Sun Nov 29, 2026 · ~210 hrs · *the bedrock — do NOT rush*

**Goal:** real mathematical maturity — hand-derive backprop, implement micrograd from memory, whiteboard attention with the √dₖ rationale. The Phase-1 self-test (Week 10) is the gate that switches the Threads on.

**Blocks:** **A** 06:00–08:00 (hardest new theory, derive by hand) · **B** 08:30–10:30 (build in code) · **Evening** 20:30–22:30 (re-watch, Anki, glossary, plan tomorrow). **Weekend** = ~10 h buffer (spillover/revision/rest — protect one full day off).
**Threads (start Week 10, inside the existing blocks):** **T** = DSA+C++ (first ~45 min of Block B) · **M** = implement-from-scratch (Block B) · **R** = reproduce-a-paper + write + apply (Evening, ~2 h/wk). *(Phase 0 is already covered in the Quarter-1 schedule.)*

---

### Week 3 · Sep 21–27 — Linear Algebra I
| Day | Block A (06–08) | Block B (08:30–10:30) | Evening (20:30–22:30) |
|---|---|---|---|
| Mon | 3B1B LA ch 1–2: vectors, span, linear combinations | NumPy: vectors, linear combos, visualize | Re-watch 3B1B; Anki (LA terms); glossary |
| Tue | 3B1B: linear transformations, matmul as composition | Implement matmul + 2D transforms; plot | Anki; glossary |
| Wed | Determinants (3B1B) + MIT 18.06 L1–3 | Implement determinant; test invertibility | Anki; glossary |
| Thu | Inverse, rank, column/null space (18.06 L4–6) | Solve Ax=b; rank experiments | Anki; glossary |
| Fri | Dot product, projections (18.06 L14–15) | Implement projection onto a subspace | Whiteboard-Fri: projection formula; teach-a-junior note |
| Sat–Sun | **Buffer** — finish spillover · light review · **1 full rest day** | | |

### Week 4 · Sep 28–Oct 4 — Linear Algebra II (eigen / SVD — the crux)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Eigenvalues/eigenvectors (3B1B + 18.06 L21) | Implement eigendecomposition; verify | Anki (eigen); glossary |
| Tue | Change of basis, diagonalization (L22) | Diagonalize; power iteration | Anki |
| Wed | **SVD** (18.065) — derive from eigendecomp of AᵀA | Implement SVD; compare to numpy | Anki (SVD — triangulate if slippery) |
| Thu | Rank-r approx (Eckart–Young); PSD | Truncated-SVD image compression | Anki |
| Fri | SVD applications; PCA preview | SVD-based PCA on a toy set | Whiteboard-Fri: SVD-from-eigendecomp; why AᵀA is PSD |
| Sat–Sun | **Buffer + rest** | | |

### Week 5 · Oct 5–11 — 🛑 CONSOLIDATION (no new material, ~18 h)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon–Fri | Re-derive from memory: matmul-as-composition, determinant, SVD-from-eigendecomp, projection | Re-implement (blank file): SVD-PCA + a subspace projection | Full Anki + glossary catch-up; Fri: teach-a-junior note on SVD |
| Sat–Sun | **Rest** + optional buffer for any LA gap | | |

### Week 6 · Oct 12–18 — LA finish + Calculus / matrix-calculus I
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | PSD, quadratic forms (18.06) | Implement quadratic-form eval; check PD | Anki |
| Tue | 3B1B Calculus: derivatives, chain rule | Numeric derivative + finite-diff check | Anki |
| Wed | Partial derivatives, gradients | Gradient of a scalar field; plot | Anki |
| Thu | Matrix calculus (Parr & Howard): ∂(Wx) | Implement ∂L/∂W = δxᵀ; grad-check | Anki |
| Fri | Jacobians | Jacobian of a vector function | Whiteboard-Fri: backprop through a linear layer |
| Sat–Sun | **Buffer + rest** | | |

### Week 7 · Oct 19–25 — Matrix-calc II + Probability I
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Softmax + cross-entropy gradient (derive p−y) | Implement softmax-CE fwd/bwd; grad-check | Anki |
| Tue | Stat 110: axioms, conditional probability | Simulate conditional probability | Anki |
| Wed | Bayes' theorem | Bayes worked example in code | Anki |
| Thu | Random variables, expectation | Simulate E[X], Var[X] | Anki |
| Fri | Variance, covariance | Covariance matrix in code | Whiteboard-Fri: Bayes + softmax-CE gradient |
| Sat–Sun | **Buffer + rest** | | |

### Week 8 · Oct 26–Nov 1 — Probability II
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Distributions (Bernoulli/Binomial/Gaussian) | Sample & fit distributions | Anki |
| Tue | MLE — derive Gaussian mean/variance | Implement an MLE fit | Anki |
| Wed | Conditional expectation as projection | E[X|Y] demo in code | Anki |
| Thu | CLT (proof sketch) | CLT simulation | Anki |
| Fri | Entropy / cross-entropy / KL | Implement entropy, CE, KL | Whiteboard-Fri: E[X|Y] as projection; MLE |
| Sat–Sun | **Buffer + rest** | | |

### Week 9 · Nov 2–8 — Probability finish + NN-from-scratch I (micrograd)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Finish key probability (covariance, moments) | Wrap probability drills | Anki |
| Tue | Karpathy Z2H: micrograd (autodiff intuition) | Start micrograd (the `Value` class) | Re-watch micrograd |
| Wed | Backprop as chain rule over a graph | micrograd: the backward pass | Anki (autodiff) |
| Thu | micrograd: a tiny MLP | micrograd: build + train an MLP | Anki |
| Fri | Review autodiff | Finish + test micrograd MLP | Whiteboard-Fri: backprop through an MLP; note |
| Sat–Sun | **Buffer + rest** | | |

### Week 10 · Nov 9–15 — NN II + **THREADS BEGIN** + Phase-1 self-test
| Day | Block A | Block B (T/M + build) | Evening (+ R) |
|---|---|---|---|
| Mon | Z2H makemore (bigram → MLP LM) | **T:** NeetCode arrays/two-pointer (45m) → makemore | Re-watch; Anki; **R:** choose month-1 paper to reproduce |
| Tue | Z2H: BatchNorm internals | **M:** implement BatchNorm (45m) → makemore | Anki |
| Wed | Z2H: manual backprop ("backprop ninja") | **T:** binary search / sliding window → build | Anki |
| Thu | Attention: scaled dot-product + √dₖ | **M:** implement scaled-dot-product attention | Anki (attention) |
| Fri | **🚩 PHASE-1 SELF-TEST** (derive backprop; micrograd from memory; BatchNorm why; attention+√dₖ) | **M:** multi-head attention | **R:** start reproduce-paper |
| Sat–Sun | **Buffer + rest** (if self-test not clean → shore up here, shift a few days) | | |

### Week 11 · Nov 16–22 — Z2H finish (GPT) + imaging kickoff · threads running
| Day | Block A | Block B (T/M + build) | Evening (+ R) |
|---|---|---|---|
| Mon | Z2H: build a small GPT (blocks, pos-enc) | **T:** stacks/queues → GPT build | Imaging physics (MRI Q&A); **R:** reproduce-paper |
| Tue | GPT: attention blocks | **M:** sampler (top-k/temperature) → GPT | Anki |
| Wed | GPT: training loop | **T:** linked lists → finish GPT | Fluorescence primer (iBiology) |
| Thu | Tokenizer (BPE) | **M:** mini-BPE → tokenizer | Anki; **R:** write-up progress |
| Fri | Review the NN stack | **T:** hashing / two-sum patterns | Whiteboard-Fri: GPT data flow; note |
| Sat–Sun | **Buffer + rest** | | |

### Week 12 · Nov 23–29 — 🛑 CONSOLIDATION + light threads
| Day | Block A | Block B (+ light T) | Evening (+ R) |
|---|---|---|---|
| Mon–Fri | Re-derive (blank page): backprop, attention+√dₖ, softmax-CE gradient, SVD | Re-implement micrograd-MLP + multi-head attention; **T:** spaced review of solved problems | Anki catch-up; **R:** finish month-1 reproduce-paper + write-up; Fri: teach-a-junior note on attention |
| Sat–Sun | **Rest** + buffer | | |

**End of Phase 1 → Phase 2 (Biomedical Imaging) next.** Gate to advance: the Week-10 self-test is clean **and** you can, cold — derive SVD from eigendecomposition, hand-derive backprop, implement micrograd from memory, and whiteboard attention with √dₖ.
