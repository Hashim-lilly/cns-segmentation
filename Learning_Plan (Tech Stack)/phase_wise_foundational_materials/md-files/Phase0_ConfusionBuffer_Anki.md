# Phase 0 — Confusion Buffer & Anki Pack (Fundamentals: Stats, Classic ML & DL)
### Companion to the Phase-0 daily schedule (Week 0 ramp + Weeks 1–2). Your articulation warm-up — pass the leaving bar before you start applying. *(Expanded edition.)*

**How to use:** each day's Block-A concept → glossary line + a few cards that evening → say it aloud (Feynman) next morning. These are the fundamentals interviewers assume you own cold.

## Confusion Buffer (tuned for fundamentals)
Intuition → rigour → implement; living glossary nightly; spaced Anki 1/3/7/30 d; Feynman gate = done; stuck >30 min → switch teacher (StatQuest ↔ 3B1B ↔ a blog). **Triangulation targets:** *what a p-value/CI actually mean*, *why L1→sparsity*, *ROC vs PR under imbalance*, *bias–variance*, *data leakage*.

## Ranked hard-topics map
1. **Bias–variance & the train/val/test discipline** (leakage is the silent killer).
2. **Probability vocabulary** — p-value/CI/CLT/MLE said correctly.
3. **Regularization** — L1 vs L2, and *why*.
4. **Imbalanced-data evaluation** — ROC vs PR, precision/recall trade-offs.
5. **Trees→ensembles** — bagging vs boosting, GBM vs RF.

## Anki deck (`Q → A`)

### Deck A · Statistics & probability foundations
- **Q:** What is a p-value, precisely? → **A:** P(a test statistic at least as extreme as observed | null true). *Not* P(null true), *not* P(result due to chance).
- **Q:** What does a 95% CI mean? → **A:** Over many repetitions of the procedure, ~95% of intervals contain the true parameter — a property of the procedure, not this one interval.
- **Q:** State the CLT. → **A:** The sample-mean distribution of i.i.d. finite-variance variables → Normal as n→∞, regardless of the population shape.
- **Q:** Standard error vs standard deviation? → **A:** SD = spread of the data; SE = SD/√n = spread of the sample-mean estimate (shrinks with n).
- **Q:** Covariance vs correlation? → **A:** Covariance = joint variability (scale-dependent); correlation = covariance normalized to [−1,1] (scale-free). Both measure *linear* association only.
- **Q:** Correlation ≠ causation — why, in one word? → **A:** Confounders (a third variable driving both); also reverse causation and selection bias. Causation needs experiments or causal assumptions.
- **Q:** Frequentist vs Bayesian — one line? → **A:** Frequentist treats parameters as fixed and data as random (CIs, p-values); Bayesian treats parameters as random with a prior, updated to a posterior via Bayes.
- **Q:** Type I vs Type II error + power? → **A:** Type I = false positive (reject a true null, rate α); Type II = false negative (rate β); power = 1−β = P(detect a true effect).
- **Q:** Statistical vs practical significance? → **A:** Statistical = unlikely under the null (p<α); practical = the effect is large enough to matter. Large n can make trivial effects "significant."
- **Q:** The multiple-comparisons problem? → **A:** Testing many hypotheses inflates the chance of a false positive; correct with Bonferroni/FDR (Benjamini-Hochberg).
- **Q:** Mean vs median — robustness? → **A:** The median is robust to outliers/skew; the mean is not. Report the median for skewed distributions.

### Deck B · Estimation & inference
- **Q:** Define MLE. → **A:** θ̂ = argmax_θ p(data|θ), usually via the log-likelihood.
- **Q:** Derive the MLE mean of a Gaussian. → **A:** ℓ = −(1/2σ²)Σ(xᵢ−μ)²+c; dℓ/dμ=0 → μ̂ = mean(x).
- **Q:** MLE vs MAP? → **A:** MLE maximizes the likelihood; MAP maximizes likelihood × prior (posterior). MAP = MLE + regularization from the prior; they coincide with a flat prior.
- **Q:** Bias–variance decomposition. → **A:** E[(y−f̂)²] = Bias² + Variance + irreducible noise. High bias = underfit; high variance = overfit.
- **Q:** Why does more data cut variance but not bias? → **A:** More data stabilizes the fit (↓variance) but can't fix a too-simple model class (bias).
- **Q:** What does the bootstrap do? → **A:** Resamples the data with replacement to estimate the sampling distribution/CI of a statistic without distributional assumptions.
- **Q:** What is an unbiased estimator, and is lower-variance always better? → **A:** E[θ̂]=θ; but a slightly biased, much-lower-variance estimator can have lower total error (bias-variance again) — hence regularization.

### Deck C · Classic ML — models
- **Q:** Linear vs logistic regression? → **A:** Linear predicts a real value (MSE loss); logistic predicts a probability via the sigmoid (cross-entropy loss) for classification.
- **Q:** Why does L1 give sparsity (gradient-level)? → **A:** L1's subgradient is ±λ (constant magnitude) → pushes small weights to exactly 0; L2's 2λw shrinks proportionally, never exactly zero.
- **Q:** Ridge vs Lasso vs Elastic Net? → **A:** Ridge = L2 (shrink, keep all features); Lasso = L1 (sparse selection); Elastic Net = L1+L2 (sparse but stable with correlated features).
- **Q:** Decision-tree split criterion? → **A:** Choose the split maximizing information gain — reducing Gini impurity or entropy (classification) / variance (regression).
- **Q:** GBM vs Random Forest? → **A:** RF = parallel bagging of decorrelated deep trees, averaged (↓variance); GBM = sequential boosting of shallow trees on residuals/gradients (↓bias), stronger but more overfit-prone.
- **Q:** Bagging vs boosting? → **A:** Bagging = train on bootstrap samples in parallel, average (variance↓); boosting = train sequentially, each correcting the last (bias↓).
- **Q:** The kernel trick? → **A:** Compute inner products in a high-dim feature space via k(x,x′) without forming the features → linear methods learn nonlinear boundaries.
- **Q:** What does an SVM maximize? → **A:** The margin — distance from the boundary to the nearest points (support vectors); soft margin (C) trades margin width for violations.
- **Q:** PCA + link to SVD? → **A:** Orthogonal directions of max variance = eigenvectors of the covariance = right-singular vectors of the *centered* data matrix.
- **Q:** k-NN — trade-offs? → **A:** No training; predict by nearest neighbors; suffers the curse of dimensionality and O(n) query cost; needs feature scaling + a good distance metric.
- **Q:** Naive Bayes assumption? → **A:** Features conditionally independent given the class — often false but effective, especially for text.

### Deck D · Classic ML — training & evaluation
- **Q:** AUC-ROC vs PR-AUC — when is PR honest? → **A:** ROC-AUC = P(random positive ranked above random negative), threshold-free; under heavy imbalance ROC looks optimistic → PR-AUC reflects the rare class better.
- **Q:** Precision vs recall vs F1? → **A:** Precision = TP/(TP+FP) (purity); recall = TP/(TP+FN) (coverage); F1 = harmonic mean — use when you need a single imbalance-aware number.
- **Q:** What is data leakage (+ examples)? → **A:** Test information sneaking into training → optimistic metrics that don't generalize; examples: scaling using test stats, target leakage from future features, duplicate/near-duplicate rows across splits.
- **Q:** Why standardize/scale features? → **A:** Distance- and gradient-based methods (k-NN, SVM, linear models, NN) are dominated by large-scale features otherwise; fit the scaler on *train only*.
- **Q:** How to handle class imbalance (3 ways)? → **A:** Resampling (over/under, SMOTE), class weights in the loss, and imbalance-aware metrics/thresholds (PR-AUC, adjust the decision threshold).
- **Q:** What does k-fold cross-validation give? → **A:** A lower-variance estimate of generalization by averaging over k train/val splits (use stratified/group folds to avoid leakage).
- **Q:** One-hot vs label (ordinal) encoding? → **A:** One-hot for nominal categories (no order); label/ordinal only when categories are genuinely ordered — else you inject a false ranking.
- **Q:** Curse of dimensionality? → **A:** As dimensions grow, data becomes sparse and distances concentrate → models overfit and neighbor methods degrade; mitigate with feature selection/reduction.
- **Q:** Why do ensembles help? → **A:** Averaging decorrelated errors reduces variance (bagging) or sequentially reduces bias (boosting) — the whole beats the average member.
- **Q:** Two signs you're overfitting? → **A:** Training loss ≫ better than validation loss, and validation loss rising while training loss keeps falling.

### Deck E · Deep-learning fundamentals
- **Q:** Why do neural nets need nonlinear activations? → **A:** Without them, stacked linear layers collapse to one linear map — no added expressiveness.
- **Q:** Gradient descent in one line? → **A:** θ ← θ − η·∇_θ L; step downhill along the negative gradient, η = learning rate.
- **Q:** Batch vs epoch vs iteration? → **A:** Batch = samples per update; iteration = one update; epoch = one full pass over the data (dataset/batch iterations).
- **Q:** SGD vs full-batch GD — why SGD? → **A:** SGD updates on mini-batches → cheaper, noisier steps that escape shallow minima and generalize well; full-batch is expensive and can overfit sharp minima.
- **Q:** ReLU vs sigmoid — one ReLU advantage + one risk? → **A:** Advantage: no saturation for positive inputs → far less vanishing gradient; risk: "dead ReLUs" (stuck at 0) — mitigated by LeakyReLU/GELU + good init.
- **Q:** Dropout — what and when off? → **A:** Randomly zero activations during training to regularize (prevents co-adaptation); turned OFF at inference (scale accordingly).
- **Q:** Weight decay vs dropout? → **A:** Weight decay = L2 penalty on weights (shrinks); dropout = stochastic co-adaptation prevention — both regularize, often used together.
- **Q:** Early stopping? → **A:** Stop training when validation loss stops improving → a simple, effective regularizer against overfitting.
- **Q:** Why "deep"? (representation view) → **A:** Depth builds a hierarchy of features (edges→parts→objects), composing simple functions into complex ones far more parameter-efficiently than a shallow net.
- **Q:** What does softmax output + when to use it? → **A:** A probability distribution over classes (exponentiate + normalize); used as the final layer for multi-class classification (paired with cross-entropy).
- **Q:** If the learning rate is too high / too low? → **A:** Too high → diverges/oscillates; too low → slow, may stall; a schedule (warmup + decay) usually helps.

## Common misconceptions & traps
- **p-value ≠ P(hypothesis true)** and non-significant ≠ "no effect."
- **Correlation ≠ causation** — confounders, reverse causation, selection bias.
- **Accuracy is misleading under imbalance** — 95%-background scores 95% predicting all background; use PR-AUC/recall.
- **High training accuracy ≠ a good model** — check validation + the bias–variance picture.
- **Data leakage is everywhere** — scale on train only, split before feature engineering, watch near-duplicates and target leakage.
- **PCA/k-NN/SVM need feature scaling** — otherwise large-scale features dominate.
- **More features ≠ better** — irrelevant features add variance (curse of dimensionality).

## Glossary starter
p-value · CI · CLT · standard error · covariance/correlation · confounder · frequentist/Bayesian · Type I/II · power · multiple comparisons/FDR · MLE/MAP · bias/variance · bootstrap · unbiased estimator · linear/logistic regression · ridge/lasso/elastic-net · Gini/entropy/info-gain · bagging/boosting · GBM/RF · kernel trick · SVM margin · PCA/SVD · k-NN · naive Bayes · ROC-AUC/PR-AUC · precision/recall/F1 · data leakage · feature scaling · class imbalance/SMOTE · cross-validation (stratified/group) · one-hot/ordinal · curse of dimensionality · ensemble · overfitting · gradient descent/SGD · batch/epoch/iteration · ReLU/sigmoid/GELU · dead ReLU · dropout · weight decay · early stopping · softmax/cross-entropy · learning-rate schedule.

## Drills
**Whiteboard:** derive the MLE mean/variance of a Gaussian; explain a 95% CI; show why L1 zeros weights and L2 shrinks; draw the bias–variance trade-off; explain why AUC misleads under imbalance; list 3 leakage sources.
**Blank-file:** logistic regression + L1/L2 from scratch (watch weights zero out); ROC & PR curves on an imbalanced toy set; a bias–variance curve (poly degree sweep); a decision-tree split by info gain; a 1-neuron gradient-descent loop; k-fold CV with a stratified split.

## Leaving bar (cold, no notes)
Explain — p-value, CI, CLT, MLE (+derive Gaussian mean), MLE vs MAP, bias–variance, ROC/AUC vs PR, precision/recall/F1, why L1→sparsity, ridge vs lasso, GBM vs RF, bagging vs boosting, kernel trick, PCA↔SVD, data leakage + fixes, class-imbalance handling, gradient descent + LR effects, dropout/weight-decay/early-stopping, why depth. Pass → **start Tier-0 applications.**
