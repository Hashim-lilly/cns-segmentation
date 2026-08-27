# Phase Q — Day-by-Day Schedule · Quantum Computing + Quantum Machine Learning
### Weeks 76–86 · Mon Feb 14 → Sun May 1, 2028 · ~335 hrs · off the job-critical path (interview in parallel)

**Goal:** match your hands-on VQE/ADAPT/DMET work with rigorous theory — derive VQE from the variational principle, explain barren plateaus, defend ansatz/measurement choices, and give an honest QML take (Schuld). Also grounds Phase-5D physics.

> **COI:** public molecules only (H₂/LiH/H₂O or a QM9 subset). No Lilly molecules (no keto-enol/HATU specifics), nothing from the vendor engagement.

*Blocks: **A** 06–08 (theory, derive) · **B** 08:30–10:30 (PennyLane/Qiskit coding) · **Evening** 20:30–22:30 (read, Anki). Weekend = buffer + rest. Threads light: **T/R** kept minimal — **you're interviewing in parallel**, so protect energy; pull Phase Q forward/interleave if a QML role appears earlier.*

---

### Week 76 · Feb 14–20 — Intuition entry + group theory / symmetry
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Quantum Country (spaced primer); MIT 8.04 intro | Qubit states in code (numpy) | Anki |
| Tue | Complex vector spaces; bra-ket | Bloch-sphere visualization | Anki |
| Wed | Groups & representations | Symmetry operations in code | Anki |
| Thu | Lie groups; link to equivariance (GDL) | Rotation reps | **R:** light applications |
| Fri | Why symmetry underlies operators & equivariant nets | Review | Whiteboard-Fri: groups/representations |
| Sat–Sun | **Buffer + rest** | | |

### Week 77 · Feb 21–27 — Quantum-mechanics math
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Hilbert spaces; inner products | State vectors in code | Anki |
| Tue | Operators; Hermitian/unitary | Operator algebra | Anki |
| Wed | Tensor products; multi-qubit states | Tensor two qubits | Anki |
| Thu | Measurement; eigenvalues as outcomes | Measurement simulation | **R:** applications |
| Fri | Pure vs mixed; density matrix | Density-matrix code | Whiteboard-Fri: tensor two qubit states; unitarity |
| Sat–Sun | **Buffer + rest** | | |

### Week 78 · Feb 28–Mar 5 — QC fundamentals: Basics of Quantum Information (Watrous/IBM)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Single-qubit gates | Qiskit: single-qubit circuits | Anki |
| Tue | Multi-qubit gates; entanglement | Bell states | Anki |
| Wed | Circuit model | Build/measure circuits | Anki |
| Thu | Superdense coding / teleportation | Implement teleportation | **R:** applications |
| Fri | Review | Circuit drills | Whiteboard-Fri: entanglement; a 2-qubit gate |
| Sat–Sun | **Buffer + rest** | | |

### Week 79 · Mar 6–12 — Fundamentals of Quantum Algorithms
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Deutsch-Jozsa; oracles | Qiskit: DJ | Anki |
| Tue | Quantum Fourier Transform | Implement QFT | Anki |
| Wed | Phase estimation | Implement QPE | Anki |
| Thu | Grover (concept) | Grover demo | **R:** applications |
| Fri | What each algorithm teaches | Review | Whiteboard-Fri: QPE walkthrough |
| Sat–Sun | **Buffer + rest** | | |

### Week 80 · Mar 13–19 — General formulation + error-correction intro
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Density matrices; quantum channels | Channel simulation | Anki |
| Tue | POVMs; noise models | Noise in Qiskit | Anki |
| Wed | Error-correction basics (bit/phase flip) | 3-qubit code demo | Anki |
| Thu | Error mitigation (near-term) | Mitigation demo | **R:** applications |
| Fri | Why noise matters for VQE | Review | Whiteboard-Fri: a channel + POVM |
| Sat–Sun | **Buffer + rest** | | |

### Week 81 · Mar 20–26 — Variational algorithms: VQE / ADAPT-VQE
| Day | Block A | Block B (PennyLane) | Evening |
|---|---|---|---|
| Mon | Variational principle (derive) | VQE on H₂ (UCCSD) | Anki |
| Tue | Ansätze; parameter-shift rule | Parameter-shift gradients | Anki |
| Wed | ADAPT-VQE (operator pool) | ADAPT-VQE on LiH | Anki |
| Thu | Optimizers for VQE | Compare optimizers | **R:** applications |
| Fri | Why ADAPT is parameter-efficient | Review | Whiteboard-Fri: derive VQE from the variational principle |
| Sat–Sun | **Buffer + rest** | | |

### Week 82 · Mar 27–Apr 2 — Variational: DMET-VQE + QAOA + measurement
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | DMET fragmentation | DMET-VQE on a small system | Anki |
| Tue | Classical-quantum interface | Fragment + embed | Anki |
| Wed | QAOA (combinatorial) | QAOA on Max-Cut | Anki |
| Thu | Measurement: Pauli grouping, shot budget | Grouping demo | **R:** applications |
| Fri | When DMET is worth the overhead | Review | Whiteboard-Fri: the classical-quantum interface |
| Sat–Sun | **Buffer + rest** | | |

### Week 83 · Apr 3–9 — QML theory (kernels, barren plateaus, Schuld critique)
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | QML as kernel methods | A quantum-kernel demo | Anki |
| Tue | Barren plateaus (math) | Gradient-variance experiment | Anki |
| Wed | Mitigations (init, local cost) | Local-cost demo | Anki |
| Thu | Schuld "taking stock" critique | Review claims critically | **R:** applications |
| Fri | The honest hype-vs-promise take | Review | Whiteboard-Fri: a barren-plateau argument |
| Sat–Sun | **Buffer + rest** | | |

### Week 84 · Apr 10–16 — QML theory cont + encodings
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Data encodings (angle/amplitude/basis) | Encoding demos | Anki |
| Tue | Expressibility & entangling capability | Expressibility experiment | Anki |
| Wed | Is quantum advantage the right goal? | Review | Anki |
| Thu | Trainability vs expressibility trade-off | PQC experiment | **R:** applications |
| Fri | QML limits, honestly | Review | Whiteboard-Fri: QML-as-kernels |
| Sat–Sun | **Buffer + rest** | | |

### Week 85 · Apr 17–23 — Quantum chemistry on quantum computers
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Qiskit summer school: chemistry track | End-to-end small-molecule pipeline | Anki |
| Tue | Basis sets on QC; active spaces | Active-space selection | Anki |
| Wed | Sample-based/Krylov diagonalization (SQD) | SQD demo | Anki |
| Thu | Benchmarking vs classical CCSD | Compare to CCSD/6-31G | **R:** applications |
| Fri | Where QC chemistry stands today | Review | Whiteboard-Fri: VQE vs QAOA — when each |
| Sat–Sun | **Buffer + rest** | | |

### Week 86 · Apr 24–May 1 — 🎯 Capstone Q + self-test
| Day | Block A | Block B | Evening (+ R) |
|---|---|---|---|
| Mon | Design the capstone (public molecule) | VQE + UCCSD on H₂/LiH/H₂O | Anki |
| Tue | Add ADAPT-VQE + DMET-VQE | Run + benchmark vs CCSD | Log |
| Wed | Barren-plateau analysis | Ansatz/optimizer/measurement ablation | Anki |
| Thu | Write-up (4 pages) | Defend choices + noise mitigation | **R:** post |
| Fri | **🚩 PHASE-Q SELF-TEST** (derive VQE from variational principle; parameter-shift rule; barren-plateau argument; VQE vs QAOA; honest QML take; DMET interface) | Final polish | **▶ QML-branch applications** |
| Sat–Sun | **Buffer + rest** · *Deliverable:* Capstone Q (public molecules) · **full-depth plan complete ~mid-2028** | | |

**End of Phase Q — the full plan is complete. QML career branch open; continue interviewing across all tiers.**
