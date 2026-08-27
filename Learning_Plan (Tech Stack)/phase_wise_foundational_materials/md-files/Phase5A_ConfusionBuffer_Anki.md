# Phase 5A — Confusion Buffer & Anki Pack (Geometric & Molecular ML Core)
### Companion to the Phase-5 daily schedule, sub-phase 5A (Weeks 43–45). Part of the Molecular-ML / Drug-Discovery track (5A–5F).

**How to use:** Block-A concept → cards + glossary → re-derive next morning. *(The shared Confusion-Buffer discipline — intuition→rigour→implement, nightly glossary, spaced Anki 1/3/7/30 d, Feynman gate, switch-teacher-when-stuck — applies to all of 5A–5F; each pack lists only its sub-phase triangulation targets.)*
**5A triangulation targets:** the *message-passing update*, *invariance vs equivariance*, the *Weisfeiler-Lehman bound*, and *when a fingerprint still beats a GNN*.

## Anki deck (`Q → A`)

### Graph ML & representations
- **Q:** Write the message-passing update. → **A:** For each node: mᵢ = AGG_{j∈N(i)} φ(hᵢ, hⱼ, eᵢⱼ); hᵢ ← UPDATE(hᵢ, mᵢ). AGG is permutation-invariant (sum/mean/max).
- **Q:** Why must the readout be permutation-invariant? → **A:** A molecule has no canonical node order; a graph-level prediction must be identical under any relabeling of atoms → use sum/mean/attention pooling.
- **Q:** What is the Weisfeiler-Lehman expressiveness bound? → **A:** Standard message-passing GNNs are at most as powerful as the 1-WL graph-isomorphism test — they can't distinguish some non-isomorphic graphs (e.g., certain regular graphs).
- **Q:** Fingerprints vs descriptors vs learned embeddings? → **A:** Fingerprints (ECFP/Morgan) = hashed substructure bits; descriptors = physicochemical numbers (logP, TPSA…); learned = GNN/transformer embeddings trained end-to-end.
- **Q:** Why scaffold splits, not random? → **A:** Random splits leak near-duplicate analogs across train/test → optimistic metrics; scaffold (or temporal) splits test generalization to *new chemotypes*, which is what matters in discovery.

### Message-passing for molecules
- **Q:** What is a D-MPNN (Chemprop) and its key idea? → **A:** A directed message-passing net whose messages live on *directed bonds/edges* (not atoms); this reduces "tottering" (messages bouncing back along the edge they came from).
- **Q:** Directed vs undirected messages — why directed? → **A:** Directed edge messages avoid immediately sending a node's own message back to itself, giving cleaner information propagation.
- **Q:** Over-smoothing in deep GNNs + 2 fixes? → **A:** Many layers make all node features converge to similar values (indistinguishable); fixes: residual/jumping-knowledge connections, fewer layers, normalization, or graph-transformer global attention.
- **Q:** How does a graph transformer inject structure into attention? → **A:** Via structural/positional encodings (degree/centrality, shortest-path or spatial encodings, edge features added to attention logits) so global attention still respects graph topology.

### 3D / equivariant networks
- **Q:** Invariance vs equivariance (formal)? → **A:** Invariant: f(g·x)=f(x); equivariant: f(g·x)=g·f(x) — the output transforms the same way as the input under group action g.
- **Q:** Why does E(3)-equivariance matter for molecules? → **A:** Energies are invariant and forces are equivariant to rotation/translation/reflection; baking this in means the model doesn't waste data relearning symmetry and stays consistent under pose.
- **Q:** SchNet — core mechanism? → **A:** Continuous-filter convolutions over interatomic distances (radial basis) → rotation-invariant energy prediction.
- **Q:** DimeNet — what does it add? → **A:** Directional message passing using distances *and* angles between triplets → captures angular/directional information SchNet misses.
- **Q:** How does EGNN update coordinates equivariantly? → **A:** Positions are updated by a sum of relative-position vectors (xᵢ−xⱼ) weighted by learned scalar functions of invariant features → the update rotates/translates with the input, no spherical harmonics needed.
- **Q:** NequIP / MACE — what are they for? → **A:** E(3)-equivariant message-passing interatomic potentials (tensor-product messages over spherical harmonics) that reach near-DFT accuracy for energies/forces with high data-efficiency.
- **Q:** When does a 2D-graph model beat a 3D one (and vice versa)? → **A:** 2D wins when 3D conformers are unavailable/unreliable or the property is topology-driven; 3D-equivariant wins for geometry-dependent properties (energies, forces, binding).

## Common misconceptions & traps
- **"A GNN always beats fingerprints."** On small, well-defined endpoints a fingerprint + gradient-boosting model is a strong, hard-to-beat baseline — always report it.
- **"Random train/test split is fine."** For molecules it leaks; use scaffold/temporal splits.
- **"More message-passing layers = better."** Over-smoothing degrades deep GNNs.
- **"3D always beats 2D."** Only when conformers are good and the target is geometry-dependent.

## Glossary starter
message passing · aggregation/readout (permutation-invariant) · Weisfeiler-Lehman · ECFP/Morgan fingerprint · descriptor · scaffold split · D-MPNN (Chemprop) · tottering · over-smoothing · graph transformer · invariance/equivariance · E(3) · SchNet · DimeNet · EGNN · NequIP/MACE · interatomic potential.

## Drills
**Whiteboard:** the message-passing update + prove readout permutation-invariance; state the WL bound; define invariance vs equivariance; explain EGNN's equivariant coordinate update.
**Blank-file:** a message-passing GNN layer from scratch (forward+backward, grad-checked); a fingerprint+GBM baseline on a TDC task; an EGNN/SchNet on a QM9 property with an equivariance-check (rotate input → output transforms).

## Leaving line (5A)
Cold: message passing vs convolution; what a D-MPNN buys over a fingerprint and when the fingerprint wins; the WL bound + over-smoothing; invariance vs equivariance and why 3D-equivariance matters for energies/forces.
