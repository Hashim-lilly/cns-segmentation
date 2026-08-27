# Phase 6 — Day-by-Day Schedule · Generative & Agentic AI (scientific / chemistry agents)
### Weeks 58–63 · Mon Oct 11 → Sun Nov 21, 2027 · ~155 hrs

**Goal:** own modern LLM adaptation, multimodal, and agentic systems — reframed toward science. Wrap a model as an MCP tool, build a multi-step research/critique agent, and evaluate agents rigorously (reinforces your critic-agent doc-review project).

*Blocks: **A** 06–08 · **B** 08:30–10:30 (build + threads) · **Evening** 20:30–22:30 (read, Anki, R). Weekend = buffer + rest. Threads: **T** DSA · **M** implement · **R** research/apply.*

---

### Week 58 · Oct 11–17 — Advanced LLMs / adaptation
| Day | Block A | Block B (T/M) | Evening (+ R) |
|---|---|---|---|
| Mon | LoRA (low-rank adaptation) | **M:** implement a LoRA layer | Anki |
| Tue | QLoRA; quantized fine-tuning | **T:** DP → a QLoRA fine-tune run | Anki |
| Wed | PEFT trade-offs | **M:** compare full vs LoRA fine-tune | Anki |
| Thu | When to fine-tune vs prompt vs RAG | **T:** graphs → decision matrix | **R:** reproduce-paper |
| Fri | Adapter/prefix methods | **T:** timed set | Whiteboard-Fri: LoRA mechanics + when to use |
| Sat–Sun | **Buffer + rest** | | |

### Week 59 · Oct 18–24 — Alignment (RLHF / DPO) + fine-tuning
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | RLHF pipeline (reward model, PPO) | **M:** a toy reward model | Anki |
| Tue | DPO (direct preference optimization) | **T:** DP → a DPO objective | Anki |
| Wed | RLHF vs DPO trade-offs | **M:** a small preference fine-tune | Anki |
| Thu | Instruction tuning; data quality | **T:** graphs → dataset prep | **R:** write-up |
| Fri | Alignment failure modes | **T:** timed set | Whiteboard-Fri: RLHF vs DPO |
| Sat–Sun | **Buffer + rest** | | |

### Week 60 · Oct 25–31 — Multimodal / VLM
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | CLIP (contrastive image-text) | **M:** a small CLIP retrieval | Anki |
| Tue | LLaVA (visual instruction) | **T:** DP → run a VLM on samples | Anki |
| Wed | Flamingo (cross-attention fusion) | **M:** image+text embedding align | Anki |
| Thu | Multimodal for imaging+text / graph+text | **T:** graphs → a molecular-graph+text demo | **R:** reproduce-paper |
| Fri | VLM failure modes in science | **T:** timed set | Whiteboard-Fri: how CLIP aligns modalities |
| Sat–Sun | **Buffer + rest** | | |

### Week 61 · Nov 1–7 — Agents + MCP
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Agent patterns (Anthropic) | **M:** a routing agent | Anki |
| Tue | Tool use; function calling | **T:** DP → tool schema + calls | Anki |
| Wed | MCP (expose models/tools) | **M:** wrap your property/UQ model as an MCP tool | Anki |
| Thu | Orchestrator-worker; evaluator-optimizer (critic loop) | **T:** graphs → a critic/verifier loop | **R:** write-up |
| Fri | Memory & state (LangGraph) | **T:** timed set | Whiteboard-Fri: MCP tool-exposure end-to-end |
| Sat–Sun | **Buffer + rest** | | |

### Week 62 · Nov 8–14 — Agent/critique evaluation + 🎯 Capstone 4 build
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Evaluating agents (task success) | **M:** build the assistant skeleton (RAG+tools) | Anki |
| Tue | Evaluating a critic (faithfulness/self-consistency) | **T:** DP → add the critic loop | Anki |
| Wed | Grounding & citations over literature | **M:** wire PMC-OA / arXiv retrieval | Anki |
| Thu | Failure analysis | **T:** graphs → eval harness for the agent | **R:** reproduce-paper |
| Fri | When an agent beats a single call | **T:** timed set | Whiteboard-Fri: how you'd evaluate the critic |
| Sat–Sun | **Buffer + rest** | | |

### Week 63 · Nov 15–21 — 🎯 Capstone 4 finish + self-test
| Day | Block A | Block B | Evening (+ R) |
|---|---|---|---|
| Mon | Polish the agent design | Finish the MCP scientific assistant (public data only) | Anki |
| Tue | Add molecular-property + UQ tool (ties 5E/5F) | Integrate + test | Anki |
| Wed | Eval + write-up | Run evals; document | **R:** write-up |
| Thu | Repo hardening | README + demo | **R:** post |
| Fri | **🚩 PHASE-6 SELF-TEST** (design an agent with a critic loop + how to evaluate the critic; MCP end-to-end; RLHF vs DPO; when an agent beats a single call) | Final polish | **▶ Tier-2 applications continue** |
| Sat–Sun | **Buffer + rest** · *Deliverable:* Capstone 4 (MCP scientific assistant, public) | | |

**End of Phase 6 → Phase 7 (Computer Vision expert) next.**
