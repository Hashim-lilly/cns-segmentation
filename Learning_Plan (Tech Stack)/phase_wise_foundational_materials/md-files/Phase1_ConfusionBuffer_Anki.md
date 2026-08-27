# Phase 1 — Confusion Buffer & Anki Pack (Mathematics + NNs from First Principles)
### Companion to the Phase-1 daily schedule (Weeks 3–12). The bedrock; the leaving bar here is the gate that switches the Threads on. *(Expanded edition.)*

**How to use:** Block-A derivation → cards + glossary that night → re-derive from a blank page next morning. Prefer explain/derive cards over recall.

## Confusion Buffer (tuned for the math build)
Intuition (3B1B) → rigour (MIT/Strang, Stat 110) → derive by hand; glossary nightly; spaced Anki 1/3/7/30 d; Feynman gate = done; stuck >30 min → switch teacher. **Triangulation targets:** *SVD-from-eigendecomposition*, *backprop*, *matrix calculus*, *Lagrange multipliers*, *the reparameterization-free MLE/Bayes glue*, *attention + √dₖ*.

## Ranked hard-topics map
1. **SVD & the spectral picture** — derive from eigendecomposition; PSD; low-rank.
2. **Backprop & autodiff** — the δ-recursion, hand-derived.
3. **Matrix calculus** — gradients of linear layers + softmax-CE.
4. **Probability glue** — Bayes, conditional expectation, MLE↔NLL↔CE.
5. **Attention** — scaled-dot-product + √dₖ + multi-head.

## Anki deck (`Q → A`)

### Deck A · Linear algebra
- **Q:** State the SVD of a real m×n A. → **A:** A = UΣVᵀ; U,V orthogonal; Σ diagonal σ₁≥σ₂≥…≥0.
- **Q:** SVD ↔ eigendecomposition? → **A:** V = eigenvectors of AᵀA; U = eigenvectors of AAᵀ; σᵢ = √(eigenvalues of AᵀA).
- **Q:** Why is AᵀA PSD? → **A:** xᵀAᵀAx = ‖Ax‖² ≥ 0 ∀x → eigenvalues ≥ 0.
- **Q:** Eckart–Young (best rank-r)? → **A:** A_r = Σ_{i≤r} σᵢuᵢvᵢᵀ minimizes ‖A−A_r‖ in Frobenius & spectral norm.
- **Q:** Geometric meaning of the determinant? → **A:** Signed volume-scaling of the map; det=0 ⇒ collapses dimension ⇒ non-invertible.
- **Q:** Rank, column space, null space? → **A:** Rank = # independent columns = dim(column space); null space = {x : Ax=0}; rank + nullity = #columns (rank-nullity).
- **Q:** Orthogonal / orthonormal — why nice? → **A:** Orthonormal columns give QᵀQ=I → Qᵀ=Q⁻¹, norm-preserving, numerically stable; the basis for QR and stable projections.
- **Q:** Spectral theorem (symmetric matrices)? → **A:** A real symmetric matrix has real eigenvalues and an orthonormal eigenbasis: A = QΛQᵀ.
- **Q:** Least-squares projection of b onto col(A)? → **A:** x̂=(AᵀA)⁻¹Aᵀb; p=Ax̂; residual b−p ⟂ col(A) (the normal equations).
- **Q:** Positive-definite vs semidefinite + why care? → **A:** PD: xᵀMx>0 ∀x≠0 (eigenvalues>0); PSD: ≥0. A PD Hessian ⇒ local strict min / local convexity; covariance matrices are PSD.
- **Q:** Condition number — what it tells you? → **A:** σ_max/σ_min; large ⇒ ill-conditioned (sensitive to noise, slow/zig-zag gradient descent).
- **Q:** Quadratic form + its gradient? → **A:** f(x)=xᵀAx; ∇f = (A+Aᵀ)x = 2Ax for symmetric A.
- **Q:** Pseudo-inverse — what for? → **A:** A⁺ = VΣ⁺Uᵀ gives the min-norm least-squares solution when A isn't invertible/square.

### Deck B · Calculus & matrix calculus
- **Q:** Gradient vs directional derivative? → **A:** The gradient points in the steepest-ascent direction; the directional derivative along u is ∇f·u.
- **Q:** ∂L/∂W for z=Wx+b (shape)? → **A:** δxᵀ where δ=∂L/∂z; same shape as W (outer product). ∂L/∂b = δ.
- **Q:** ∂(softmax+CE)/∂logits? → **A:** p − y (predicted probs minus one-hot) — the clean result behind softmax-CE.
- **Q:** Jacobian vs Hessian? → **A:** Jacobian = matrix of first partials (vector→vector); Hessian = matrix of second partials (curvature of a scalar function).
- **Q:** Chain rule for f(g(x))? → **A:** df/dx = f′(g(x))·g′(x); for vectors, multiply Jacobians right-to-left.
- **Q:** Taylor expansion (to 2nd order) — why useful? → **A:** f(x+Δ) ≈ f(x) + ∇fᵀΔ + ½ΔᵀHΔ; the basis for Newton's method and analyzing curvature/optimization.
- **Q:** Convexity — the Hessian test? → **A:** f is convex iff its Hessian is PSD everywhere → any local min is global (why convex problems are "easy").
- **Q:** Lagrange multipliers — the idea? → **A:** To optimize f subject to g=0, set ∇f = λ∇g (gradients parallel at the optimum); λ = sensitivity of the optimum to the constraint.

### Deck C · Probability
- **Q:** Bayes' theorem + terms? → **A:** P(θ|D)=P(D|θ)P(θ)/P(D): posterior ∝ likelihood × prior / evidence.
- **Q:** Law of total probability / expectation? → **A:** P(A)=Σ P(A|Bᵢ)P(Bᵢ); E[X]=E[E[X|Y]] (tower rule).
- **Q:** E[X|Y] as a projection? → **A:** The best mean-square predictor of X from Y = orthogonal projection of X onto functions of Y.
- **Q:** E and Var of a sum of independent vars? → **A:** E[ΣXᵢ]=ΣE (always); Var[ΣXᵢ]=ΣVar (only if independent/uncorrelated).
- **Q:** Independent vs uncorrelated? → **A:** Independent ⇒ uncorrelated; the converse fails in general (uncorrelated just means zero linear association). For jointly Gaussian, they're equivalent.
- **Q:** Name 4 distributions + a use? → **A:** Bernoulli/Binomial (counts of successes), Poisson (rare-event counts), Gaussian (CLT/noise), Exponential (waiting times).
- **Q:** Key Gaussian facts? → **A:** Fully specified by mean+covariance; linear combos of Gaussians are Gaussian; marginals/conditionals are Gaussian; max-entropy for fixed variance.
- **Q:** What is a covariance matrix + a property? → **A:** Σ_{ij}=Cov(Xᵢ,Xⱼ); symmetric PSD; its eigenvectors are the PCA axes.
- **Q:** Why is minimizing NLL = MLE = minimizing CE? → **A:** NLL=−Σlog p(x|θ) → maximizing log-likelihood (MLE); for a categorical target it *is* the cross-entropy between the label and the prediction.
- **Q:** Conjugate prior — why care? → **A:** A prior whose posterior stays in the same family (e.g., Beta–Bernoulli) → closed-form Bayesian updates.

### Deck D · Neural networks from scratch
- **Q:** Backprop in one sentence? → **A:** The chain rule applied layer-by-layer, reusing cached forward activations to get ∂L/∂param for all params in one backward pass.
- **Q:** Backward one layer: δ_{l-1} from δ_l? → **A:** δ_{l-1} = (W_lᵀ δ_l) ⊙ f′(z_{l-1}).
- **Q:** What is a computational graph? → **A:** A DAG of elementary ops; forward computes values, backward applies the chain rule in reverse-topological order accumulating gradients.
- **Q:** Reverse-mode autodiff — cost? → **A:** Exact gradients at ≈ one forward pass of compute (memory for cached activations) → why it's used for scalar losses of many params.
- **Q:** Vanishing gradients — cause + 2 fixes? → **A:** Repeated ×small activation-derivatives shrinks gradients with depth; fixes: ReLU-family + residual connections + normalization + good init.
- **Q:** Exploding gradients — fix? → **A:** Gradient clipping (by norm) + proper init + normalization.
- **Q:** Xavier vs He init — why? → **A:** Scale init so activation/gradient variance is preserved across layers — Xavier (tanh/linear), He (ReLU); prevents exploding/vanishing at the start.
- **Q:** Activation comparison (sigmoid/tanh/ReLU/GELU)? → **A:** sigmoid/tanh saturate (vanishing gradients); ReLU is fast, non-saturating but can die; GELU/Swish are smooth ReLU variants used in transformers.
- **Q:** Why residual connections? → **A:** y=x+F(x) gives gradients a direct path (identity) → trains very deep nets and eases optimization (mitigates degradation).
- **Q:** What does BatchNorm do / why help / when break? → **A:** Normalize a feature over the batch then scale/shift (γ,β); smooths the landscape, enables higher LR; breaks with tiny batches, needs running stats at eval.
- **Q:** Softmax + cross-entropy — why combined numerically? → **A:** Combine into log-softmax/logsumexp for stability (avoid overflow/log(0)); gradient wrt logits = p − y.

### Deck E · Attention (end of phase)
- **Q:** Scaled-dot-product attention? → **A:** softmax(QKᵀ/√dₖ)V.
- **Q:** Why ÷√dₖ? → **A:** QKᵀ has variance ~dₖ; rescaling to unit variance stops softmax saturating into near-one-hot (which vanishes gradients).
- **Q:** Q, K, V intuition? → **A:** Query = what I seek; Key = what I offer; Value = what I pass if matched; output = similarity-weighted sum of values.
- **Q:** Why multi-head? → **A:** Parallel lower-dim heads capture different subspaces/relations, concatenated — more expressive than one head.
- **Q:** Self- vs cross-attention? → **A:** Self: Q,K,V from the same sequence (intra-sequence relations); cross: Q from one sequence, K,V from another (e.g., decoder attending to encoder).

## Common misconceptions & traps
- **Backprop isn't magic** — it's the chain rule + caching; hand-derive it for a 2-layer MLP.
- **BatchNorm behaves differently at train vs eval** (batch stats vs running stats) — a classic bug.
- **Initialization matters** — poor init causes exploding/vanishing signals before training starts.
- **Depth without residuals/normalization doesn't help** — signal degrades; that's what residual/norm fix.
- **SVD is not just PCA** — PCA is SVD on the *centered* data; SVD is the general factorization.
- **Uncorrelated ≠ independent** (except jointly Gaussian).
- **A large condition number** silently slows/destabilizes gradient descent — normalize/precondition.

## Glossary starter
SVD / singular value · eigenvector/eigenvalue · rank/null space/rank-nullity · orthonormal/QR · spectral theorem · PSD/PD · Eckart–Young · determinant · projection/least-squares/normal equations · condition number · quadratic form · pseudo-inverse · gradient/directional derivative · Jacobian/Hessian · chain rule · Taylor expansion · convexity (PSD Hessian) · Lagrange multipliers · Bayes · law of total probability/expectation (tower) · conditional expectation · independent vs uncorrelated · Bernoulli/Binomial/Poisson/Gaussian/Exponential · covariance matrix · MLE/MAP/NLL/CE · conjugate prior · backprop (δ-recursion) · computational graph · autodiff (reverse mode) · vanishing/exploding gradients · gradient clipping · Xavier/He init · ReLU/GELU/dead-ReLU · residual connection · BatchNorm (train/eval) · log-softmax/logsumexp · scaled-dot-product attention · √dₖ · Q/K/V · multi-head · self/cross-attention.

## Drills
**Whiteboard:** derive SVD from the eigendecomposition of AᵀA; prove AᵀA is PSD; least-squares normal equations; ∂(softmax+CE)/∂logits = p−y; Lagrange-multiplier setup for a constrained optimum; full backprop for a 2-layer MLP; scaled-dot-product attention + √dₖ.
**Blank-file:** SVD-based PCA + truncated-SVD compression; a subspace projection; softmax+CE fwd/bwd with a numerical gradient check; **micrograd** (scalar autodiff + MLP) from memory; multi-head attention; a residual block; He-init a small net and check activation variances.

## Leaving bar (cold, no notes) → Threads T/M/R begin
Derive SVD from eigendecomposition + why AᵀA is PSD; the least-squares normal equations; ∂(softmax+CE)/∂logits; hand-derive backprop for a 2-layer MLP + implement micrograd from memory; explain vanishing/exploding gradients + fixes; why BatchNorm helps + what breaks; Bayes + E[X|Y]-as-projection + MLE↔NLL↔CE; whiteboard scaled-dot-product attention with √dₖ + multi-head.
