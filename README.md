# 🧠 Project: Adaptive Rehearsal in Continual Learning using Concept Drift Detection

![Status](https://img.shields.io/badge/status-In%20Progress-yellow)

A Major Project by **Abhinav**

---

## 1. Project Overview
Continual learning systems struggle with **catastrophic forgetting**—models lose performance on old tasks when exposed to new ones. Rehearsal methods such as **iCaRL** replay stored exemplars to retain knowledge, yet they do so at a constant rate, wasting computation when the model is already stable. Meanwhile, the data-stream mining community has mature **concept drift detectors** (e.g., **ADWIN**) that flag statistically significant performance drops.

> **Core Hypothesis**: By combining iCaRL with ADWIN we can build an *adaptive* rehearsal strategy that reacts only when forgetting is detected, preserving accuracy while reducing rehearsal cost.

> 📄 **Mid-Semester Dossier**: A comprehensive narrative covering background, methodology, experimental roadmap, and presentation outline is available in [`docs/mid-semester-review.md`](docs/mid-semester-review.md).


---

## 2. Current Idea Review

| Component | Strengths | Limitations |
| --- | --- | --- |
| **iCaRL (Incremental Classifier and Representation Learning)** | Strong baseline for class-incremental learning, exemplar management, balanced fine-tuning. | Fixed rehearsal schedule increases training time and energy use even during stable phases. |
| **ADWIN (Adaptive Windowing) Drift Detector** | Provides theoretical guarantees on detecting distribution change in streaming settings, efficient online updates. | Requires performance signals (loss/accuracy) and can trigger false alarms without calibration. |

### Why the Hybrid Makes Sense
1. **Complementary Signals** – iCaRL supplies exemplar buffers and evaluation hooks; ADWIN monitors metrics to decide when rehearsal is necessary.
2. **Resource Awareness** – Adaptive triggering reduces redundant rehearsal epochs, aligning with compute-constrained deployment scenarios.
3. **Clear Research Questions** – How sensitive must the detector be? What latency is acceptable between drift detection and recovery? Can dynamic rehearsal match iCaRL accuracy with fewer updates?

---

## 3. Semester Learning & Development Plan

| Timeline (2025) | Focus | Outcomes |
| --- | --- | --- |
| **Weeks 1-2 (Aug)** | Deep dive into continual learning fundamentals; reproduce simple rehearsal baselines (e.g., ER, GEM). | Literature notes, baseline scripts, evaluation harness (Split CIFAR-10/100). |
| **Weeks 3-5 (Sept)** | Implement and validate vanilla iCaRL; document architecture, exemplar selection, incremental training loop. | Verified iCaRL baseline with metrics + reproducible notebook. |
| **Weeks 6-8 (Oct)** | Study ADWIN / drift detection; build lightweight monitor pipeline over rehearsal metrics; run ablation to calibrate thresholds. | Modular ADWIN monitor, experiments on synthetic drift streams. |
| **Weeks 9-11 (Nov)** | Integrate adaptive trigger into iCaRL loop; design experiments comparing constant vs adaptive rehearsal. | Prototype "Smart Rehearsal" model, initial comparison plots. |
| **Weeks 12-14 (Dec)** | Optimise, evaluate on additional datasets (e.g., Split Tiny-ImageNet); prepare visualisations and documentation. | Final metrics table, compute usage analysis, polished charts. |
| **Week 15 (Jan)** | Finalise report, presentation, code clean-up, reproducibility checklist. | Submission-ready artefacts and presentation deck. |

---

## 4. Working Flow (Planned System Architecture)
1. **Data Stream Intake** → Tasks arrive sequentially (e.g., Split CIFAR-10).
2. **Feature Extractor & Classifier (iCaRL backbone)** → Train incrementally on current task with exemplar rehearsal buffer.
3. **Performance Monitor** → Maintain validation accuracy / loss on exemplar set.
4. **ADWIN Drift Detector** → Consume performance stream, raise alarms on significant drops.
5. **Adaptive Rehearsal Trigger**
   - If *no drift*: continue lightweight rehearsal (minimal or zero replay).
   - If *drift detected*: launch focused rehearsal burst (balanced fine-tuning + exemplar updates).
6. **Metrics Logger** → Track accuracy, forgetting measure, rehearsal time, compute cost.
7. **Analysis & Visualisation** → Compare adaptive vs static rehearsal across tasks.

This flow will be implemented modularly so each component can be evaluated independently during development.

---

## 5. Evaluation Deliverables

### Mid-Semester Evaluation (Idea Validation)
- **Problem Statement & Motivation** – Written summary of catastrophic forgetting and inefficiencies in static rehearsal.
- **Literature Review Snapshot** – Two-page synthesis of rehearsal and drift-detection approaches with identified gap.
- **System Design Draft** – Architecture diagram & flow description (Section 4) plus success criteria.
- **Baseline Plan** – Experimental setup for iCaRL baseline, datasets, and evaluation metrics.
- **Expected Outcomes** – Hypothesised benefits (accuracy parity, reduced rehearsal cost) and risk analysis.

### End-Semester Evaluation (Project Completion)
- **Implementation Report** – Detailed methodology, modular design, and final algorithm.
- **Experimental Results** – Tables/plots comparing adaptive vs static rehearsal, including compute metrics.
- **Ablation & Sensitivity Analysis** – Impact of detector parameters, rehearsal burst size, buffer limits.
- **Repository Deliverables** – Clean codebase, reproducible scripts, README updates, final presentation slides.
- **Reflection & Future Work** – Lessons learned, limitations, and potential extensions (e.g., other detectors, task-free settings).

---

## 6. References & Resources
1. Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). *iCaRL: Incremental Classifier and Representation Learning*. CVPR.
2. van de Ven, G. M., & Tolias, A. S. (2019). *Three scenarios for continual learning*. arXiv:1904.07734.
3. Bifet, A., & Gavalda, R. (2007). *Learning from time-changing data with adaptive windowing*. SDM.
4. Montiel, J., Read, J., Bifet, A., & Abdessalem, T. (2021). *River: Machine learning for streaming data in Python*. JMLR.

---

> This README will evolve alongside the project. Upcoming additions include experiment trackers, visual dashboards, and links to evaluation reports once submitted.
