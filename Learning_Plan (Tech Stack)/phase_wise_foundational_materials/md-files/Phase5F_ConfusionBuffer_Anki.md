# Phase 5F — Confusion Buffer & Anki Pack (Applied Cheminformatics & Discovery ML)
### Companion to Phase-5 sub-phase 5F (Weeks 56–57). Part of the Molecular-ML track (5A–5F).

**How to use:** as 5A. **5F triangulation targets:** *MCTS retrosynthesis*, *why DEL counts are noisy and how enrichment modeling denoises them*, *multitask vs negative transfer (MMoE)*, and *what federated learning trades off*.

## Anki deck (`Q → A`)
- **Q:** How does MCTS retrosynthesis search? → **A:** Monte-Carlo Tree Search over reaction templates: **selection** (pick promising nodes via UCT) → **expansion** (apply an expansion policy to propose precursors) → **rollout/evaluation** → **backup** (update values); goal = reach purchasable building blocks (the "stock").
- **Q:** Single-step vs multi-step retrosynthesis? → **A:** Single-step = predict immediate precursors for one target; multi-step = recursively search a full synthetic route down to available building blocks.
- **Q:** Why is reaction-condition prediction hard? → **A:** It's multi-label (reagent/solvent/catalyst/temperature), data is noisy/imbalanced (USPTO/ORD), and many valid condition sets exist → evaluate with top-k accuracy, not exact match.
- **Q:** What is a DNA-encoded library (DEL) and why are counts noisy? → **A:** Millions of compounds tagged with DNA barcodes screened in one pool; the readout is sequencing *counts*, corrupted by PCR bias, truncates, and non-specific binding → raw counts ≠ affinity.
- **Q:** How does enrichment modeling denoise DEL data? → **A:** Model the count-generating process (e.g., compare selection vs control, ZINB/Poisson models, or ML on building-block features) to estimate true enrichment/binding signal from noisy counts.
- **Q:** Multitask learning — benefit and risk? → **A:** Benefit = shared representation transfers across related endpoints (data-efficient); risk = **negative transfer** — unrelated/conflicting tasks hurt each other through the shared trunk.
- **Q:** How does MMoE mitigate negative transfer? → **A:** Multi-gate Mixture-of-Experts: shared expert networks + a per-task gating that lets each task use a different expert mix → tasks share where helpful, diverge where not.
- **Q:** Contrastive representation of binding — idea? → **A:** Pull together representations of binders to the same target (or matched pairs), push apart non-binders → embeddings where distance reflects binding similarity.
- **Q:** What does federated learning trade off? → **A:** Trains across sites without pooling raw data (privacy) but pays in communication cost, statistical heterogeneity (non-IID data across sites), and residual leakage risk (gradients can leak information).
- **Q:** The active-learning virtual-screening loop? → **A:** Score a large library with a cheap ML surrogate → dock/validate the most promising (or most uncertain) → add results to the training set → retrain → repeat; picks the next compounds to evaluate to maximize info/hits per cost.
- **Q:** Why enumerate building blocks / synthons? → **A:** Constrains generation/search to *make-on-demand* chemistry, ensuring proposed molecules are synthetically accessible from a real catalog.

## Common misconceptions & traps
- **"Multitask always helps."** Negative transfer is real; use MMoE/gating or task grouping, and check per-task metrics.
- **"DEL counts equal affinity."** They're noisy proxies; enrichment modeling/denoising is required.
- **"Federated learning = free privacy."** Gradients can leak; non-IID data and communication cost are real obstacles.
- **"A high virtual-screening score is a hit."** It's a triage signal; experimental (or FEP/MD) validation is still needed.

## Glossary starter
retrosynthesis (single/multi-step) · MCTS (selection/expansion/rollout/backup, UCT) · building-block stock · reaction-condition prediction (multi-label, top-k) · DEL / barcode counts · enrichment modeling / denoising · multitask learning · negative transfer · MMoE (multi-gate MoE) · contrastive binding representation · federated learning (non-IID, communication, leakage) · active-learning virtual screening · synthon/building-block enumeration.

## Drills
**Whiteboard:** the MCTS retrosynthesis loop; why DEL counts are noisy + how to denoise; multitask vs negative transfer + how MMoE helps; the federated-learning trade-offs.
**Blank-file:** an AiZynthFinder run on public targets; a reaction-condition model with top-k accuracy; a DEL enrichment model on BELKA; a Flower federated multi-endpoint ADMET demo; an active-learning VS loop (surrogate + dock).

## Leaving line (5F)
Cold: MCTS retrosynthesis; DEL count noise + enrichment denoising; multitask negative transfer and MMoE; federated-learning trade-offs; the active-learning virtual-screening loop.
