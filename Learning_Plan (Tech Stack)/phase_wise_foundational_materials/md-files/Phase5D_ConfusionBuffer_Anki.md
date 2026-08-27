# Phase 5D — Confusion Buffer & Anki Pack (Physics-Based Structure & Simulation)
### Companion to Phase-5 sub-phase 5D (Weeks 50–53). Part of the Molecular-ML track (5A–5F). *The DESRES-core sub-phase; bridges your quantum-chemistry work.*

**How to use:** as 5A. **5D triangulation targets:** the *accuracy/cost ladder*, the *thermodynamic cycle behind relative FEP*, *what a docking score really approximates*, and *how an ML potential reaches DFT quality*.

## Anki deck (`Q → A`)

### Conformers & the ladder
- **Q:** The molecular-simulation accuracy/cost ladder (cheap→accurate)? → **A:** MMFF (force field) → GFN2-xTB (semi-empirical) → ML interatomic potential (MACE-OFF/ANI) → DFT → (for binding) FEP → MD sampling. Cost rises with accuracy.
- **Q:** What is ETKDGv3? → **A:** RDKit's knowledge-based distance-geometry method for generating initial 3D conformers (using torsion/experimental preferences), usually followed by force-field optimization.
- **Q:** Why is MMFF energy ordering unreliable; the fix? → **A:** Classical force fields miss subtle electronic effects → wrong relative conformer energies; re-rank the top conformers with DFT (e.g., ωB97X-D/def2-SVP).
- **Q:** COV vs MAT (conformer metrics)? → **A:** Coverage (COV) = recall — fraction of reference conformers matched within an RMSD threshold; Matching (MAT) = precision — average RMSD of the closest generated conformer to each reference.
- **Q:** Why symmetry-corrected RMSD? → **A:** Molecular symmetry means atoms can be permuted equivalently; RMSD must account for that or it over-penalizes chemically identical poses.

### Docking
- **Q:** What does a docking score approximate — and why not affinity? → **A:** An empirical/knowledge-based estimate of binding pose quality; it's a rough proxy, not a rigorous free energy — good for ranking/enrichment, poor as an absolute ΔG.
- **Q:** Pose RMSD vs enrichment — why both? → **A:** Pose RMSD checks geometric accuracy of the predicted binding pose; enrichment (EF/BEDROC) checks whether actives rank above decoys in a screen — a method can be right on one, wrong on the other.
- **Q:** gnina vs classical Vina? → **A:** gnina adds a CNN-based scoring function on top of docking; Vina uses an empirical score + search. gnina can improve pose selection/scoring.
- **Q:** What does DiffDock change? → **A:** Treats docking as generative pose prediction via diffusion over ligand translations/rotations/torsions — instead of exhaustive search + scoring.

### Free energy & MD
- **Q:** Thermodynamic cycle behind relative binding FEP? → **A:** Compute the *alchemical* free energy of mutating ligand A→B both bound and free; the binding-ΔΔG = ΔG_bound − ΔG_free — errors in the endpoints cancel, giving accurate *relative* affinities.
- **Q:** Why is FEP accurate but expensive, and where does it fail? → **A:** It samples the true statistical ensemble with explicit solvent (accurate), but needs long MD + many λ-windows (expensive); fails with poor force fields, insufficient sampling, or large perturbations (scaffold hops).
- **Q:** What is a force field? → **A:** A parameterized potential-energy function (bonds/angles/torsions/electrostatics/vdW); fast but an approximation — its quality bounds MD/FEP accuracy.
- **Q:** Why does MD need thermostats/barostats + PBC? → **A:** Thermostat holds temperature (NVT), barostat holds pressure (NPT), periodic boundary conditions mimic bulk solvent without surface artifacts.
- **Q:** How does an ML interatomic potential (MACE-OFF/ANI) reach DFT quality at MM cost — and its limit? → **A:** Trained on DFT energies/forces, an equivariant net predicts them at a fraction of the cost; limit = transferability — it can be unreliable outside its training chemical space.
- **Q:** DFT essentials (functional + basis set)? → **A:** DFT approximates electronic energy via an exchange-correlation *functional* (e.g., ωB97X-D, with dispersion) evaluated in a *basis set* (e.g., def2-SVP); choice trades accuracy vs cost.
- **Q:** How does this ladder connect to your VQE/quantum work? → **A:** VQE/DMET-VQE target the electronic-structure energy that DFT approximates — the quantum-chemistry rung sits at the accurate/expensive top of the same ladder.

## Common misconceptions & traps
- **"Docking score = binding affinity."** No — it's a ranking proxy, not a rigorous ΔG.
- **"MD/FEP is ground truth."** It's a *sampled* approximation, bounded by the force field and sampling time.
- **"An ML potential is universal."** Transferability is limited to its training distribution.
- **"More conformers is always better."** Returns diminish; what matters is covering the low-energy ensemble within a sensible energy window.

## Glossary starter
force field (MMFF) · GFN2-xTB · ML interatomic potential (MACE-OFF/ANI) · DFT (functional/basis set/dispersion) · ETKDGv3 · conformer ensemble · COV/MAT · symmetry-corrected RMSD · docking (Vina/gnina/DiffDock) · pose RMSD · enrichment (EF/BEDROC) · FEP/ABFEP · thermodynamic cycle · alchemical · MD · thermostat/barostat/PBC · accuracy/cost ladder.

## Drills
**Whiteboard:** the full accuracy/cost ladder + where each rung breaks; the relative-FEP thermodynamic cycle; why a docking score isn't ΔG; COV vs MAT.
**Blank-file:** ETKDG+MMFF conformers + Butina clustering + COV/MAT vs GEOM; a Vina docking run + enrichment; a short OpenMM MD + MDAnalysis RMSD; DFT single-points to re-rank conformers.

## Leaving line (5D)
Cold: the accuracy/cost ladder and each rung's failure mode; the thermodynamic cycle behind relative FEP; what a docking score approximates (and pose vs enrichment); how an ML potential reaches DFT quality and its transferability limit; the link to VQE/DMET.
