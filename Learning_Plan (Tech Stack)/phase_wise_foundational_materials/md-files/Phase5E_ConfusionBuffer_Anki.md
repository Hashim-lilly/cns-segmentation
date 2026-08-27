# Phase 5E — Confusion Buffer & Anki Pack (Uncertainty Quantification & Probabilistic ML)
### Companion to Phase-5 sub-phase 5E (Weeks 54–55). Part of the Molecular-ML track (5A–5F). *A stated squad priority; cross-cutting.*

**How to use:** as 5A. **5E triangulation targets:** *why split conformal gives valid coverage without distributional assumptions* (and what breaks it), the *GP posterior + what the kernel encodes*, and *why deep ensembles are often the strongest practical UQ*.

## Anki deck (`Q → A`)
- **Q:** Aleatoric vs epistemic uncertainty? → **A:** Aleatoric = irreducible noise in the data (measurement/label noise); epistemic = model uncertainty from limited data/knowledge (reducible with more data). UQ methods differ in which they capture.
- **Q:** Split (inductive) conformal prediction — how? → **A:** On a held-out calibration set, compute nonconformity scores (e.g., residuals); the (1−α) quantile sets the prediction-interval width so that new points are covered ≥1−α of the time.
- **Q:** Why does conformal give *valid* coverage without distributional assumptions? → **A:** It only needs **exchangeability** of calibration + test data; the coverage guarantee is a finite-sample, distribution-free property of the quantile construction.
- **Q:** What breaks conformal's guarantee for molecules? → **A:** Distribution shift that breaks exchangeability — scaffold/temporal shift (new chemotypes) means calibration ≠ test distribution → coverage degrades. Use group/Mondrian or shift-adaptive conformal.
- **Q:** Coverage vs efficiency? → **A:** Coverage = fraction of intervals containing the truth (should hit 1−α); efficiency = interval width (narrower is better). A trivial wide interval has coverage but no utility.
- **Q:** GP regression posterior — what do you get? → **A:** A predictive **mean** and **variance** in closed form; the variance grows away from training data → principled epistemic uncertainty.
- **Q:** What does the GP kernel encode? → **A:** The assumed similarity/smoothness structure (e.g., RBF length-scale = how fast the function varies); it defines the prior over functions.
- **Q:** Why SVGP (sparse variational GP)? → **A:** Exact GPs are O(n³); SVGP uses m≪n inducing points + variational inference → scalable, mini-batchable calibrated regression.
- **Q:** Deep ensembles — method + why strong? → **A:** Train N independently-initialized nets, average predictions; disagreement estimates epistemic uncertainty. Often the best-calibrated practical UQ, at N× cost.
- **Q:** MC-dropout as UQ? → **A:** Keep dropout on at inference, average N stochastic passes; an approximate Bayesian posterior — cheaper than ensembles but usually less well-calibrated.
- **Q:** ECE / reliability diagram? → **A:** ECE = weighted mean |accuracy − confidence| across confidence bins; the reliability diagram plots accuracy vs confidence per bin. Low ECE + diagonal diagram = calibrated.
- **Q:** Temperature scaling? → **A:** Divide logits by a learned scalar T (fit on validation) to calibrate softmax confidences post-hoc — fixes over/under-confidence without changing accuracy.
- **Q:** Calibration vs coverage — different things? → **A:** Calibration = predicted probabilities match empirical frequencies (classification); coverage = intervals contain the truth at the stated rate (regression/conformal). Related goals, different guarantees.

## Common misconceptions & traps
- **"High confidence = correct."** Only if calibrated — check ECE/reliability.
- **"Conformal needs a Gaussian/known distribution."** No — it's distribution-free; it needs *exchangeability* (which molecular scaffold shift can break).
- **"One UQ method is universally best."** Deep ensembles are strong but costly; conformal wraps *any* model with coverage; GPs give smooth uncertainty but scale poorly (→ SVGP). Pick by constraints.
- **"UQ is a nice-to-have."** For decision-making (which compounds to make/test) calibrated uncertainty is the deliverable, not an add-on.

## Glossary starter
aleatoric/epistemic · conformal prediction (split/inductive) · nonconformity score · exchangeability · coverage/efficiency · Mondrian/group conformal · GP (mean/variance/kernel) · length-scale · SVGP/inducing points · deep ensemble · MC-dropout · ECE/reliability diagram · temperature scaling · calibration vs coverage.

## Drills
**Whiteboard:** why split conformal gives valid coverage + what breaks it; the GP posterior + kernel meaning; conformal vs GP vs ensemble trade-offs.
**Blank-file:** add split-conformal intervals to your 5A model + an empirical-coverage plot; an SVGP (GPyTorch) on an ADMET set; a four-way UQ comparison (conformal / GP / MC-dropout / ensemble) with coverage + ECE.

## Leaving line (5E)
Cold: why split conformal is valid without distributional assumptions and how scaffold shift breaks exchangeability; the GP posterior + what the kernel encodes; why ensembles are often the best practical UQ; ECE vs coverage.
