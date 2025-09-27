---
title: "End-Semester Report"
author: "Abhinav"
date: "December 2025"
---

# Smart Rehearsal for Continual Learning
## End-Semester Project Report

> **Purpose.** This document provides the narrative backbone for the end-semester submission.  It combines the problem framing, system design, comparative insights across code phases, and storytelling cues for the final viva voce.

---

## Executive Summary
- **Problem.** Catastrophic forgetting undermines incremental learning systems; static rehearsal expends compute even when the model is stable.
- **Proposed Solution.** Couple exemplar-based rehearsal (iCaRL) with an ADWIN drift monitor to trigger intensive replay only when performance drops.
- **Key Evidence.** Phase-by-phase experimentation demonstrates that adaptive rehearsal maintains accuracy while reducing replay cost, culminating in a modular smart pipeline.
- **Outcome.** Provide a reproducible toolkit (scripts, comparison suite, reporting guides) and a narrative suitable for both the written report and oral defence.

---

## 1. Introduction
### 1.1 Motivation
Continual learning agents deployed on streaming data must preserve earlier knowledge without full retraining. Traditional replay schedules iterate over exemplar buffers at fixed intervals, creating unnecessary compute overhead.

### 1.2 Research Gap
Existing rehearsal baselines lack sensitivity to real-time performance changes. Concept-drift detectors from the data-stream community offer a mechanism to react only when metrics deteriorate, yet they are rarely integrated with deep rehearsal systems.

### 1.3 Objectives
1. Design a modular pipeline that inserts a drift-aware trigger into the rehearsal loop.
2. Empirically compare static and adaptive strategies across incremental-learning benchmarks.
3. Deliver professional documentation (code guides, figures, appendices) for the end-semester evaluation.

---

## 2. System Overview
### 2.1 Phase-by-Phase Code Artefacts
| Phase | Script | Core Idea | Outputs |
| --- | --- | --- | --- |
| Phase 1 | `experiments/phase1/rehearsal_baseline.py` | Fixed-rate rehearsal baseline with exemplar buffer tracking. | Accuracy CSV, rehearsal statistics, optional accuracy plot. |
| Phase 2 | `experiments/phase2/adaptive_rehearsal.py` | ADWIN-triggered rehearsal bursts with JSON logs of drift events. | Metrics JSON, drift event log, optional accuracy plot. |
| Phase 3 | `experiments/phase3/smart_rehearsal_pipeline.py` | Modular smart pipeline (config dataclasses, telemetry, fallbacks). | JSONL event stream, final metrics snapshot, buffer utilisation chart. |
| Aggregation | `experiments/end_semester/comparison_suite.py` | Consolidates phase artefacts into Markdown/JSON reports. | `comparison_report.md`, `comparison_report.json`. |

### 2.2 Architecture Highlights
- **Backbone Learner.** iCaRL-style encoder/classifier with exemplar management.
- **Monitoring Layer.** Validation pipeline over stored exemplars and held-out task shards.
- **Drift Detector.** ADWIN (or fallback moving average) monitors accuracy streams.
- **Adaptive Manager.** Launches rehearsal bursts, updates buffers, and logs telemetry when drift is detected.
- **Reporting Stack.** Unified comparison suite transforms raw artefacts into report-ready tables.

---

## 3. Methodology
1. **Baseline Reproduction.** Validate fixed-rate rehearsal on SplitMNIST / Split CIFAR to establish reference accuracy and forgetting metrics.
2. **Drift Integration.** Introduce ADWIN monitoring, calibrate thresholds with synthetic drifts, and log event timelines.
3. **Smart Pipeline Assembly.** Package configuration, training loop, and telemetry into a reusable module for late-semester experiments.
4. **Aggregation & Analysis.** Run the comparison suite to produce a unified Markdown report for direct inclusion in LaTeX.

---

## 4. Experimental Design
### 4.1 Datasets
- SplitMNIST (for quick iteration and demonstrable forgetting curves).
- Split CIFAR-10 / CIFAR-100 (longer streams for the final comparative study).
- Optional: Tiny-ImageNet for stretch goals.

### 4.2 Metrics
- **Average Accuracy** after each task.
- **Forgetting** per task (max accuracy drop).
- **Replay Cost** (number of rehearsal steps, wall-clock time if available).
- **Drift Responsiveness** (time between drift detection and recovery).

### 4.3 Ablations
- Detector sensitivity (`delta`, window size).
- Buffer size trade-offs.
- Rehearsal burst length vs. accuracy recovery.

---

## 5. Comparative Analysis Blueprint
Use the outputs collected by the comparison suite to populate this section in the final report.

### 5.1 Quantitative Summary
Insert the Markdown table from `outputs/end_semester/comparison_report.md`. Highlight:
- Accuracy trends across phases.
- Reduction in rehearsal steps for adaptive phases.
- Drift response metrics (detection count, recovery time).

### 5.2 Visual Evidence
Recommended figures:
1. **Accuracy vs. Task Plot** – Phase 1 vs Phase 2 vs Phase 3.
2. **Rehearsal Cost Histogram** – Replay steps comparison.
3. **Drift Timeline** – Annotated Phase 2/3 drift events.

### 5.3 Qualitative Insights
- Discuss scenarios where adaptive rehearsal skips redundant replay.
- Explain failure cases (missed drifts, false positives) and mitigation strategies.
- Connect observed behaviour to theoretical expectations from ADWIN.

---

## 6. Implementation Notes for the Report Appendix
- Provide pseudo-code for the adaptive rehearsal loop (Phase 3).
- Document configuration defaults (`buffer_size`, `detector_delta`, `evaluation_interval`).
- Include a reproducibility checklist referencing command-line invocations.
- Summarise hardware used for final experiments.

---

## 7. Future Work
- Explore alternative detectors (DDM, Page-Hinkley).
- Extend to task-free continual learning setups.
- Integrate hyperparameter sweeps for automatic detector calibration.
- Deploy on resource-constrained hardware to quantify efficiency gains.

---

## 8. References
1. Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). *iCaRL: Incremental Classifier and Representation Learning*. CVPR.
2. Bifet, A., & Gavalda, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing*. SDM.
3. van de Ven, G. M., & Tolias, A. S. (2019). *Three Scenarios for Continual Learning*. arXiv:1904.07734.
4. Montiel, J., Read, J., Bifet, A., & Abdessalem, T. (2021). *River: Machine Learning for Streaming Data in Python*. JMLR.
5. Lopez-Paz, D., & Ranzato, M. (2017). *Gradient Episodic Memory for Continual Learning*. NeurIPS.

---

> **Presentation Tip.** Open the report with the executive summary, then transition to the comparison table and highlight the adaptive gains before diving into implementation specifics.
