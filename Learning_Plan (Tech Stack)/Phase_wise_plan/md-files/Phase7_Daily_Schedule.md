# Phase 7 — Day-by-Day Schedule · Computer Vision Expert (classical / geometric / 3D / edge)
### Weeks 64–69 · Mon Nov 22, 2027 → Sun Jan 2, 2028 · ~175 hrs

**Goal:** round out into a broad CV expert (video already done in 2V) with real deployment/edge skill — multi-view geometry, optimized C++/TensorRT inference, 3D/neural rendering.

*Blocks: **A** 06–08 · **B** 08:30–10:30 (build + threads) · **Evening** 20:30–22:30 (read, Anki, R). Weekend = buffer + rest. Threads: **T** DSA · **M** implement · **R** research/apply. (Holiday weeks 68 lighter.)*

---

### Week 64 · Nov 22–28 — Classical & geometric CV
| Day | Block A | Block B (T/M) | Evening (+ R) |
|---|---|---|---|
| Mon | FPCV (Nayar): image formation, filtering | **M:** edge/corner detectors from scratch | Anki |
| Tue | Pinhole camera; intrinsics/extrinsics | **T:** DP → camera calibration | Anki |
| Wed | Epipolar geometry; fundamental matrix | **M:** estimate F from correspondences | Anki |
| Thu | Features: SIFT (scale/rotation invariance) | **T:** graphs → SIFT matching | **R:** reproduce-paper |
| Fri | Homography; RANSAC | **T:** timed set | Whiteboard-Fri: pinhole model + epipolar constraint |
| Sat–Sun | **Buffer + rest** | | |

### Week 65 · Nov 29–Dec 5 — SfM / SLAM
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | Structure-from-Motion pipeline | **M:** run COLMAP on an image set | Anki |
| Tue | Bundle adjustment (concept) | **T:** DP → triangulation | Anki |
| Wed | Visual odometry; ORB-SLAM3 | **M:** ORB feature tracking | Anki |
| Thu | Stereo & depth; Open3D point clouds | **T:** graphs → point-cloud ops | **R:** write-up |
| Fri | Loop closure; drift | **T:** timed set | Whiteboard-Fri: the SfM pipeline |
| Sat–Sun | **Buffer + rest** | | |

### Week 66 · Dec 6–12 — Modern DL vision breadth
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | EECS 498: detection foundations | **M:** an anchor-based detector head | Anki |
| Tue | Two-stage vs one-stage; anchor-free | **T:** DP → NMS from scratch | Anki |
| Wed | Detectron2 / MMDetection | **M:** fine-tune a detector | Anki |
| Thu | YOLO family; segmentation heads | **T:** graphs → run YOLO | **R:** reproduce-paper |
| Fri | Metrics: mAP, IoU | **T:** timed set | Whiteboard-Fri: two-stage vs one-stage; NMS |
| Sat–Sun | **Buffer + rest** | | |

### Week 67 · Dec 13–19 — Deployment & optimization for edge
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon | ONNX export; graph optimization | **M:** export a model to ONNX | Anki |
| Tue | TensorRT; INT8 calibration | **T:** DP → quantize + benchmark | Anki |
| Wed | CUDA basics (PMPP) | **M:** a simple CUDA kernel | Anki |
| Thu | Triton / DeepStream serving | **T:** graphs → serve via Triton | **R:** write-up |
| Fri | Latency vs accuracy trade-offs | **T:** timed set | Whiteboard-Fri: what TensorRT does; INT8 calibration |
| Sat–Sun | **Buffer + rest** | | |

### Week 68 · Dec 20–26 — 🛑 CONSOLIDATION-light (holidays) + 3D vision start
| Day | Block A | Block B | Evening |
|---|---|---|---|
| Mon–Wed | Re-derive: epipolar geometry, NMS, quantization; NeRF intuition | Re-implement (blank file) 1–2 CV primitives; **T:** spaced review | Anki catch-up |
| Thu–Fri | NeRF (volume rendering) basics | Run a small nerfstudio scene; **R:** write-up | 3DGS overview |
| Sat–Sun | **Rest** (holidays) | | |

### Week 69 · Dec 27–Jan 2 — 3D vision + 🎯 Capstone 5 + self-test
| Day | Block A | Block B | Evening (+ R) |
|---|---|---|---|
| Mon | 3D Gaussian Splatting | Build Capstone 5 (SfM/VO on public set **or** TensorRT-optimize a Capstone-1/2 model) | Anki |
| Tue | Point-cloud registration (Open3D) | Continue capstone; **T:** 1 timed set | Log results |
| Wed | Real-time C++ inference pipeline | Finish + latency/accuracy report | **R:** write-up |
| Thu | Review CV breadth | Polish repo + README | **R:** post |
| Fri | **🚩 PHASE-7 SELF-TEST** (pinhole + epipolar; SIFT invariance; two-stage vs one-stage; what TensorRT does; port a PyTorch CV model to real-time C++) | Final polish | **▶ applications continue** |
| Sat–Sun | **Buffer + rest** · *Deliverable:* Capstone 5 (deployment/3D, public) | | |

**End of Phase 7 → Phase 8 (Interview crescendo) next.**
