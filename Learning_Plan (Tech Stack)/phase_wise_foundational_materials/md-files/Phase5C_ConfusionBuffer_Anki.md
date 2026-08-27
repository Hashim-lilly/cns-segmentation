# Phase 5C — Confusion Buffer & Anki Pack (Generative Chemistry)
### Companion to Phase-5 sub-phase 5C (Weeks 48–49). Part of the Molecular-ML track (5A–5F).

**How to use:** as 5A. **5C triangulation targets:** the *GFlowNet objective and why it yields reward-proportional diversity*, *equivariant diffusion over 3D coordinates*, and the *validity/novelty/diversity* metrics.

## Anki deck (`Q → A`)
- **Q:** Define validity / uniqueness / novelty / diversity. → **A:** Validity = fraction chemically valid; uniqueness = fraction non-duplicate in the sample; novelty = fraction not in the training set; diversity = average pairwise distance (e.g., Tanimoto) among generated molecules.
- **Q:** Why is "maximize reward" a bad generative objective? → **A:** It collapses to a few high-reward modes (mode collapse); hit-finding needs *diverse* high-reward candidates, not one.
- **Q:** GFlowNet — what does it learn? → **A:** A policy that builds objects step-by-step so the probability of sampling an object is proportional to its reward R(x) (not argmax) → naturally diverse high-reward samples.
- **Q:** GFlowNet training objective (trajectory balance)? → **A:** Enforce flow consistency so that, for each complete trajectory, the forward-policy flow equals the backward flow scaled by R(x); minimizing the trajectory-balance loss makes terminal sampling probability ∝ R(x).
- **Q:** GFlowNet vs RL vs VAE? → **A:** RL maximizes expected reward → mode-seeking; VAE learns a latent distribution of the *data* (limited novelty/optimization); GFlowNet samples ∝ reward → diverse, optimizable de-novo design.
- **Q:** Equivariant diffusion (EDM) for 3D molecules — idea? → **A:** A diffusion model that denoises atom coordinates (and types) with an E(3)-equivariant network, so generated 3D structures respect rotation/translation symmetry.
- **Q:** GeoDiff / torsional diffusion — what problem? → **A:** Generate/refine 3D conformers by diffusing over coordinates (GeoDiff) or over torsion angles (torsional diffusion — lower-dimensional, physically meaningful).
- **Q:** VAE for molecules (JT-VAE) + a failure mode? → **A:** Encodes molecules (as junction trees + graphs) to a latent space to decode valid molecules; failure = posterior collapse (latent ignored) and limited novelty.
- **Q:** What is a property reward (examples)? → **A:** A scalar guiding generation — QED (drug-likeness), SA-score (synthesizability), or a predicted-activity/docking proxy; often a weighted combination.
- **Q:** Why include a synthesizability constraint? → **A:** High-reward but unmakeable molecules are useless; SA-score / retrosynthesis feasibility keeps outputs actionable.
- **Q:** MOSES / GuacaMol — what are they? → **A:** Standardized benchmarks/metrics for molecular generation (validity, novelty, diversity, distribution similarity, goal-directed optimization).

## Common misconceptions & traps
- **"Highest reward = best generator."** Diversity + novelty + synthesizability matter as much as reward.
- **"Valid = useful."** A valid molecule can be trivial, known, or unmakeable — check novelty and SA.
- **"GFlowNet is just RL."** It's flow-matching: it samples *proportional* to reward (diverse), not the argmax (mode-collapsed).
- **"3D generation is solved."** Geometry validity, chirality, and strain remain hard; DFT/force-field checks are still needed.

## Glossary starter
validity/uniqueness/novelty/diversity · mode collapse · GFlowNet · trajectory balance / flow matching · reward-proportional sampling · equivariant diffusion (EDM) · GeoDiff / torsional diffusion · JT-VAE · posterior collapse · QED · SA-score · MOSES/GuacaMol.

## Drills
**Whiteboard:** the GFlowNet objective + why it yields reward-proportional diversity; GFlowNet vs RL vs diffusion; define the four generation metrics.
**Blank-file:** a metrics+reward module (validity/novelty/diversity + QED/SA); a GFlowNet on a small fragment env with a property reward; a small equivariant/3D generation run + geometry-validity check.

## Leaving line (5C)
Cold: the GFlowNet objective and the reward-proportional-diversity argument (vs RL mode-collapse); how equivariant diffusion respects symmetry; validity/novelty/diversity/synthesizability definitions.
