# Phase 3 — Confusion Buffer & Anki Pack (Modern DL Theory + Foundation Architectures)
### Companion to the Phase-3 daily schedule (Weeks 31–36). This is the phase interviewers probe hardest ("derive attention", "explain the diffusion loss") — the cards are written to make those answers automatic.

**How to use:** Block-A derivation → cards + glossary that night → re-derive next morning from a blank page. The **leaving bar** (bottom) is the gate to Phase 4.

---

## Part 1 — Confusion Buffer (tuned for DL theory)
1. **Derive, don't memorize.** Every card here has an underlying derivation — do it on paper until the card is a *summary*, not a fact.
2. **Two views for the hard objects.** Attention (algebraic ↔ retrieval intuition); diffusion (variational ELBO ↔ score-matching). Hold both.
3. **Spaced Anki** 1/3/7/30 d; **glossary** nightly.
4. **Feynman gate = done.**
5. **Unblock:** stuck >30 min → switch teacher (3B1B ↔ CS25 ↔ Annotated Transformer ↔ Lilian Weng ↔ Yang Song).
6. **Triangulation targets** (what usually doesn't click): **√dₖ scaling**, **RoPE as relative position**, **pre-LN vs post-LN**, the **diffusion loss both ways**, and **classifier-free guidance**.
7. **Consolidation:** re-derive attention + the diffusion loss + backprop weekly.
8. **Understanding-gated advance.**

## Part 2 — Hard-topics map (ranked)
1. **Attention end-to-end** — derive scaled-dot-product, √dₖ rationale, multi-head as one matmul, param count.
2. **Diffusion** — forward/reverse, the ε-loss, and the variational↔score-matching equivalence.
3. **Optimization dynamics** — momentum, Adam's two moments, AdamW's decoupled decay, warmup.
4. **Positional encodings** — sinusoidal/learned/RoPE/ALiBi and which extrapolate.
5. **Normalization placement** — LayerNorm-not-BatchNorm; pre-LN vs post-LN.
6. **Information theory glue** — CE-from-KL, NLL=MLE, MI in InfoNCE.
7. **Generative alternatives** — VAE ELBO + reparameterization; normalizing flows; why VAEs blur.
8. **Efficiency** — FlashAttention (IO-aware), GQA/MQA (KV-cache), MoE (sparse), SwiGLU.

## Part 3 — Anki deck (copy in; `Q → A`)

### Deck A · Optimization dynamics
- **Q:** Why does momentum accelerate SGD? → **A:** It accumulates an EMA of past gradients — damping oscillation across high-curvature directions and building speed along consistent low-curvature ones.
- **Q:** What do Adam's two moments do? → **A:** m = EMA of gradients (direction); v = EMA of squared gradients (per-parameter adaptive step via ÷√v). Bias-correction fixes early underestimate.
- **Q:** AdamW vs Adam? → **A:** AdamW applies weight decay directly to weights (decoupled from the adaptive step) → correct L2 regularization; plain Adam's "L2-in-gradient" interacts wrongly with ÷√v.
- **Q:** Why learning-rate warmup? → **A:** Early moment/gradient estimates are noisy and activations unstable; a ramped small LR avoids destabilizing updates (esp. Adam + LayerNorm + large batch).
- **Q:** Non-convex — why does DL still train? → **A:** Over-parameterized nets have many good, connected minima and benign landscapes; SGD noise + good init/normalization find generalizing low-loss basins.
- **Q:** What does the condition number tell you? → **A:** Ratio of max/min Hessian eigenvalues; high = ill-conditioned (slow zig-zag GD); normalization/Adam/preconditioning reduce its effect.

### Deck B · Information theory
- **Q:** Derive cross-entropy from KL. → **A:** KL(p‖q)=Σp log(p/q)=−H(p)+H(p,q); minimizing H(p,q) in q (p fixed) = minimizing KL(p‖q).
- **Q:** Why is minimizing NLL = MLE? → **A:** NLL=−Σlog q(x); minimizing it maximizes the log-likelihood → MLE.
- **Q:** Mutual information? → **A:** I(X;Y)=H(X)−H(X|Y)=KL(p(x,y)‖p(x)p(y)); info Y gives about X (0 iff independent).
- **Q:** MI in contrastive learning (InfoNCE)? → **A:** InfoNCE lower-bounds MI between two views; minimizing it maximizes positive-pair agreement vs negatives → representations keep shared information.

### Deck C · Transformers
- **Q:** Scaled-dot-product attention + shapes? → **A:** softmax(QKᵀ/√dₖ)V; Q,K ∈ ℝ^{n×dₖ}, V ∈ ℝ^{n×dᵥ}; scores n×n; output n×dᵥ.
- **Q:** Why ÷√dₖ? → **A:** QKᵀ entries have variance ~dₖ (sum of dₖ unit-variance products); rescaling to unit variance keeps softmax out of saturated near-one-hot regions (which vanish gradients).
- **Q:** Multi-head attention as one matmul? → **A:** Project to Q,K,V of width h·d_head, reshape to (h,n,d_head), batched attention, concat → (n,h·d_head), final linear. Heads run in parallel via reshaped matmuls.
- **Q:** Q, K, V intuition? → **A:** Query = what a token seeks; Key = what each offers; Value = what it contributes; output = similarity(Q,K)-weighted sum of V.
- **Q:** Sinusoidal vs learned vs RoPE vs ALiBi? → **A:** Sinusoidal = fixed sin/cos (okay extrapolation); learned = trained (no extrapolation); RoPE = rotate Q/K by position (relative, strong extrapolation); ALiBi = distance penalty added to scores (cheap, long-context).
- **Q:** How does RoPE encode position? → **A:** Rotates pairs of Q/K dims by an angle ∝ position; the dot product then depends on the *relative* offset (m−n).
- **Q:** Why LayerNorm not BatchNorm in transformers? → **A:** LN normalizes over features per token — independent of batch/sequence length, stable for variable lengths and small batches; BN's batch stats are unstable for sequences and leak across tokens.
- **Q:** Pre-LN vs post-LN? → **A:** Pre-LN (norm inside the residual branch) → stable gradients, trains without warmup-sensitivity; post-LN (original) can need careful warmup, less stable at depth.
- **Q:** What does FlashAttention do; why faster? → **A:** Computes *exact* attention in tiles in on-chip SRAM without materializing the n×n matrix in HBM — IO-aware, fewer memory reads/writes → faster, O(n) memory (compute still O(n²)).
- **Q:** MQA / GQA — what & why? → **A:** Multi-Query (1 shared KV head) / Grouped-Query (few KV heads) shrink the KV-cache & memory bandwidth at inference with minimal quality loss → fast long-context serving.
- **Q:** What is SwiGLU? → **A:** A gated FFN: (Swish(xW₁) ⊙ xW₃)W₂ — a GLU variant outperforming plain ReLU/GELU FFNs.
- **Q:** Mixture-of-Experts layer? → **A:** Replace the FFN with many experts + a router sending each token to top-k; scales parameters at ~constant per-token compute (sparse activation).
- **Q:** Param count of a block (d, FFN ratio r)? → **A:** Attention ≈ 4d² (Q,K,V,O); FFN ≈ 2·r·d² (up+down); + norms (~2d). Total ≈ (4 + 2r)d² per block.

### Deck D · ViT & modern ConvNets
- **Q:** Why does SSL (DINOv2/MAE) help low-label regimes? → **A:** Learns general features from unlabeled data; a small labeled set then only trains a head/fine-tune → beats from-scratch on scarce labels.
- **Q:** What did ConvNeXt show? → **A:** A pure CNN modernized with transformer-era choices (large kernels, LayerNorm, GELU, fewer activations) matches ViTs — the training recipe/scale matters more than the "family".

### Deck E · Diffusion & generative
- **Q:** DDPM forward process? → **A:** Add Gaussian noise over T steps: q(x_t|x_{t−1})=N(√(1−β_t)x_{t−1}, β_t I); closed form x_t=√(ᾱ_t)x₀+√(1−ᾱ_t)ε.
- **Q:** DDPM training loss (simple form)? → **A:** Predict the added noise: E‖ε − ε_θ(x_t,t)‖² (MSE).
- **Q:** Variational vs score-matching view? → **A:** Variational = maximize an ELBO on the reverse process (→ the noise-prediction loss); score-matching = learn ∇ₓ log p(x) at each noise level (sampling = reverse SDE/Langevin). Equivalent; ε-prediction ∝ the score.
- **Q:** What is the score function? → **A:** ∇ₓ log p(x) — points toward higher density; diffusion models it (scaled) via the noise predictor, enabling reverse sampling.
- **Q:** DDIM vs DDPM sampling? → **A:** DDIM = deterministic, non-Markovian sampler reusing the same model to generate in far fewer steps, comparable quality.
- **Q:** Classifier-free guidance? → **A:** Train conditionally + unconditionally (drop the condition sometimes); sample with ε = ε_uncond + s·(ε_cond − ε_uncond) to strengthen conditioning (s trades diversity for fidelity).
- **Q:** VAE ELBO + reparameterization? → **A:** ELBO = E_q[log p(x|z)] − KL(q(z|x)‖p(z)) (reconstruction − regularization); reparameterize z=μ+σ⊙ε (ε~N(0,I)) so gradients flow through sampling.
- **Q:** Why do VAEs blur? → **A:** Gaussian likelihood + KL regularization + posterior averaging favor mean-like reconstructions; diffusion/GANs sharpen.
- **Q:** Normalizing flows — idea + constraint? → **A:** Invertible maps with tractable Jacobian-determinants transform data↔prior; exact likelihood via change-of-variables; require invertibility + efficient log-det.

## Part 4 — Common misconceptions & traps
- **"Attention is O(1) memory."** No — naive attention is O(n²) memory; FlashAttention makes it O(n) memory (compute is still O(n²)).
- **"Adam always beats SGD."** SGD+momentum often generalizes better in vision; AdamW is standard for transformers — pick by regime.
- **"BatchNorm works in transformers."** LayerNorm is used; BN's batch stats are unstable for sequences.
- **"Diffusion learns the image distribution directly."** It learns to *denoise* / the score at each noise level; sampling integrates the reverse process.
- **"More positional-encoding parameters → better extrapolation."** Learned PEs don't extrapolate; RoPE/ALiBi do.
- **"Cross-entropy and KL are different objectives."** Minimizing CE in q = minimizing KL(p‖q) (they differ by the constant H(p)).
- **"Bigger batch → always faster convergence."** Needs LR scaling + warmup; past a point returns diminish and generalization can drop.

## Part 5 — Glossary starter
SGD / momentum / Adam / AdamW · warmup / LR schedule · condition number · convex vs non-convex · entropy / cross-entropy / KL / mutual information · InfoNCE · scaled-dot-product attention · √dₖ scaling · Q/K/V · multi-head · positional encoding (sinusoidal / learned / RoPE / ALiBi) · LayerNorm vs BatchNorm · pre-LN vs post-LN · FlashAttention · MQA / GQA / KV-cache · SwiGLU · MoE / router · ViT patch embedding · ConvNeXt · SSL (MAE / DINOv2) · DDPM (forward/reverse) · ε-loss · score function · DDIM · classifier-free guidance · VAE / ELBO / reparameterization · normalizing flow / change-of-variables.

## Part 6 — Drills
**Whiteboard (no notes):** derive scaled-dot-product attention + the √dₖ rationale; multi-head as one matmul; param-count a block from (d, r); derive CE from KL; derive the DDPM loss both ways (variational + score); explain pre-LN vs post-LN; RoPE as relative position.
**Blank-file (no AI):** SGD/Momentum/Adam from scratch on a 2D loss; multi-head attention (+ RoPE + RMSNorm + SwiGLU) as one block; a minimal DDPM (forward noising + reverse ε-loss + DDIM sampling) on a toy set; a small VAE with the reparameterization trick; KL between two Gaussians.

## Part 7 — Triangulation
- **Attention / √dₖ:** Annotated Transformer + 3B1B "Attention" + Lilian Weng.
- **Positional encodings:** the RoPE paper (arXiv 2104.09864) + a good blog derivation.
- **Diffusion:** Yang Song's score-based blog + Lilian Weng's diffusion post + the HF Diffusion course (implement one).
- **Optimization:** Ruder's overview + Boyd (convexity) + implement the optimizers.
- **Info theory:** MacKay ch 1–6.
- **VAE/flows:** CS236 lectures + implement a tiny VAE.

## Part 8 — Spaced-review & the leaving bar
**Daily** Anki + one re-derivation; **weekly** re-derive attention + the diffusion loss + backprop from blank pages; trust 1/3/7/30-day.
**Leaving bar (cold, unaided) →** whiteboard scaled-dot-product attention in ~5 min with the √dₖ rationale; multi-head as one matmul + param-count a block; derive CE from KL and state NLL=MLE; derive the diffusion loss both ways and explain the score-function link; explain LayerNorm-not-BatchNorm and pre-LN vs post-LN; compare the four positional encodings; explain classifier-free guidance; write the VAE ELBO + reparameterization.
