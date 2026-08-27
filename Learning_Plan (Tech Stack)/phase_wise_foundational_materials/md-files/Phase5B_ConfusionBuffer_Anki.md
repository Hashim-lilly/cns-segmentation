# Phase 5B — Confusion Buffer & Anki Pack (Molecular Foundation Models & SSL)
### Companion to Phase-5 sub-phase 5B (Weeks 46–47). Part of the Molecular-ML track (5A–5F).

**How to use:** as 5A. **5B triangulation targets:** the *MLM objective*, *what makes a valid molecular augmentation for contrastive SSL*, and *when pretraining actually helps*.

## Anki deck (`Q → A`)
- **Q:** Masked-language-modeling (MLM) objective for molecules? → **A:** Mask atoms/tokens (SMILES or graph nodes) and train the model to reconstruct them → learns chemical context without labels.
- **Q:** What is ChemBERTa? → **A:** A BERT-style transformer pretrained with masked-SMILES modeling on large unlabeled molecule sets, then fine-tuned for property tasks.
- **Q:** Contrastive SSL for molecules (MolCLR) — the idea? → **A:** Create two augmented views of a molecule, pull their embeddings together and push apart other molecules → representations invariant to label-preserving perturbations.
- **Q:** What is a *valid* molecular augmentation for contrastive SSL? → **A:** One that preserves the property of interest — e.g., atom/bond masking, subgraph removal that keeps identity; NOT changing the scaffold or a pharmacophore (that changes the molecule's meaning).
- **Q:** Uni-Mol — what's distinctive? → **A:** A 3D-aware SSL model pretrained on conformers (masked-atom + coordinate denoising) → captures geometry, not just topology/strings.
- **Q:** Linear-probe vs fine-tune? → **A:** Linear-probe = freeze the encoder, train a linear head (fast, tests representation quality); fine-tune = update the encoder too (more capacity, needs more labels, risks overfitting/forgetting).
- **Q:** Pretrain→finetune workflow? → **A:** Self-supervised pretrain on large unlabeled data → transfer weights → fine-tune (or linear-probe) on the small labeled downstream task.
- **Q:** Why does pretraining help in low-label regimes? → **A:** The encoder already captures general chemistry; the small labeled set only needs to learn a simple mapping → better than training from scratch on scarce labels.
- **Q:** How do you measure transfer gain honestly? → **A:** Compare pretrained-then-finetuned vs from-scratch at matched label budgets (e.g., 100/1k/10k), on scaffold splits.

## Common misconceptions & traps
- **"Pretraining always helps."** On small, well-defined endpoints a fingerprint+GBM or from-scratch GNN can match or beat a fine-tuned foundation model — measure, don't assume.
- **"Any augmentation works for contrastive SSL."** Bad augmentations (scaffold changes) destroy the signal — augmentations must be label-preserving chemically.
- **"Bigger pretrained model = better downstream."** Only if the domain matches and you have enough fine-tuning data; otherwise it overfits or underperforms simpler baselines.

## Glossary starter
self-supervised learning · MLM (masked-atom/SMILES) · ChemBERTa · contrastive learning · MolCLR · augmentation (label-preserving) · Uni-Mol (3D SSL) · linear-probe · fine-tune · transfer gain · label budget.

## Drills
**Whiteboard:** write the MLM objective; explain what makes a molecular augmentation valid; when SSL helps vs hurts.
**Blank-file:** a small masked-SMILES pretrain (loss curve + masked-token recovery); a pretrain→finetune transfer study vs from-scratch across label budgets.

## Leaving line (5B)
Cold: the MLM objective; a valid vs invalid molecular augmentation for contrastive SSL; when pretraining helps vs when a fingerprint+GBM still wins.
