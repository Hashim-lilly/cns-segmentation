# Phase 4 — Day-by-Day Schedule · Full-Stack / Production AI Systems + MLOps
### Weeks 37–42 · Mon May 17 → Sun Jun 27, 2027 · ~165 hrs

**Goal:** defend every architectural choice and design the next-gen version — LLM internals, senior RAG, MLOps + ML system design, distributed training, serving. Directly reinforces your critic-agent doc-review project.

*Blocks: **A** 06–08 (theory) · **B** 08:30–10:30 (build + threads) · **Evening** 20:30–22:30 (read, Anki, R). Weekend = buffer + rest. Threads: **T** DSA (Block B ~45m) · **M** implement (Block B) · **R** research/apply (Evening).*

---

### Week 37 · May 17–23 — LLM internals (CS336)
| Day | Block A | Block B (T/M) | Evening (+ R) |
|---|---|---|---|
| Mon | CS336: tokenization → embeddings | **T:** DP → implement a BPE tokenizer | Anki |
| Tue | Attention blocks, KV-cache | **M:** implement a KV-cache | Anki |
| Wed | Sampling: greedy/top-k/nucleus/temp | **T:** heaps → implement samplers | Anki |
| Thu | Scaling laws; compute budgeting | **M:** a tiny LM training loop | **R:** reproduce-paper |
| Fri | Efficiency: mixed precision, grad-checkpoint | **T:** timed set | Whiteboard-Fri: token→embedding→blocks→head |
| Sat–Sun | **Buffer + rest** | | |

### Week 38 · May 24–30 — LLM internals cont + RAG start
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Fine-tuning vs prompting vs RAG | **M:** a minimal retrieval loop | Anki |
| Tue | Embeddings & vector search | **T:** graphs → build an embedding index | Anki |
| Wed | Chunking strategies | **M:** chunking + retrieval eval | Anki |
| Thu | Hybrid search (BM25 + dense) | **T:** strings → hybrid retriever | **R:** write-up |
| Fri | Reranking | **T:** timed set | Whiteboard-Fri: chunk/embed/retrieve/rerank trade-offs |
| Sat–Sun | **Buffer + rest** | | |

### Week 39 · May 31–Jun 6 — RAG at senior level
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Hamel/Eugene Yan LLM patterns | **M:** query rewriting + reranker | Anki |
| Tue | Anthropic contextual retrieval | **T:** DP → contextual chunks | Anki |
| Wed | Evals: RAGAS, faithfulness | **M:** a RAGAS eval harness | Anki |
| Thu | DSPy / structured prompting | **T:** graphs → a DSPy pipeline | **R:** reproduce-paper |
| Fri | Where a critic/verifier improves RAG | **T:** timed set | Whiteboard-Fri: how to evaluate a RAG system honestly |
| Sat–Sun | **Buffer + rest** · *Deliverable:* RAG system (hybrid + rerank + RAGAS) on public docs | | |

### Week 40 · Jun 7–13 — MLOps + ML systems design
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Chip Huyen: ML systems design framing | **T:** DP → sketch a data/feature layer | Anki |
| Tue | CS329S: data & feature stores | **M:** a feature pipeline + versioning | Anki |
| Wed | Training→serving→monitoring loop | **T:** graphs → a serving stub | Anki |
| Thu | Drift detection; retraining triggers | **M:** a drift-detection check | **R:** write-up |
| Fri | Made With ML / FSDL patterns | **T:** timed set | Whiteboard-Fri: full train→serve→monitor design |
| Sat–Sun | **Buffer + rest** · *Deliverable:* an ML-system-design doc (e.g., molecular-property serving) | | |

### Week 41 · Jun 14–20 — Distributed training + inference optimization
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Data vs tensor vs pipeline parallelism | **T:** DP → a DDP toy | Raschka; Anki |
| Tue | ZeRO / FSDP sharding | **M:** gradient accumulation + AMP | Anki |
| Wed | vLLM + paged attention (run it) | **T:** graphs → serve with vLLM | Anki |
| Thu | Quantization (GPTQ/AWQ/GGUF/FP8) | **M:** quantize + benchmark latency | **R:** reproduce-paper |
| Fri | Throughput vs latency trade-offs | **T:** timed set | Whiteboard-Fri: what ZeRO shards; KV-cache speedups |
| Sat–Sun | **Buffer + rest** | | |

### Week 42 · Jun 21–27 — Eval / observability / safety + systems-design prep + self-test
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | LLM evals; LLM-as-judge pitfalls | **M:** an eval harness (+ judge) | Anki |
| Tue | Observability (LangSmith/Phoenix) | **T:** graphs → instrument the RAG system | Anki |
| Wed | Constitutional AI / safety basics | **M:** guardrail checks | Anki |
| Thu | System-design mock 1 (HCP RAG, 10k/day) | **T:** DP → design artifacts | **R:** write-up |
| Fri | **🚩 PHASE-4 SELF-TEST** (design an LLM-eval harness; a 45-min ML-system-design whiteboard; defend a serving/monitoring architecture) | **T:** timed set | System-design mock 2 (brain-seg pipeline w/ domain shift) |
| Sat–Sun | **Buffer + rest** | | |

**End of Phase 4 → Phase 5 (Molecular-ML / Drug-Discovery Track) next — the 15-week differentiator.**
