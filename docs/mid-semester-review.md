---
title: "Mid-Semester Review Report"
author: "Abhinav"
date: "September 2025"
---

# Adaptive Rehearsal for Continual Learning
## Mid-Semester Evaluation Dossier

## Abstract
Catastrophic forgetting remains a central challenge in continual learning: models trained on streaming tasks tend to lose performance on previously learned tasks. Classic rehearsal-based approaches such as iCaRL mitigate forgetting by replaying buffered exemplars, but they incur constant computational overhead, even when the model is stable. This project proposes **Smart Rehearsal**, an adaptive pipeline that couples exemplar-based rehearsal with the Adaptive Windowing (ADWIN) concept-drift detector. By triggering intensive rehearsal only when performance degradation is detected, the system aims to retain accuracy while reducing unnecessary computation. This mid-semester report documents the problem context, literature foundations, system design, progress to date, planned experiments, and the roadmap to the final submission.

---

## 1. Introduction
- **Problem Context**: Sequential data streams require models that adapt without revisiting the entire dataset. Traditional deep learning models retrained from scratch are inefficient for such scenarios.
- **Research Gap**: Rehearsal-based continual learning methods achieve competitive accuracy but treat replay frequency as fixed. This static strategy wastes resources during stable phases and may still miss fast-onset forgetting.
- **Project Hypothesis**: Monitoring validation performance with ADWIN can signal genuine forgetting events, allowing the system to trigger rehearsal bursts only when necessary, thus balancing retention and efficiency.

---

## 2. Background and Related Work
1. **Rehearsal-Based Methods**
   - *Experience Replay (ER)* maintains a memory buffer of past samples for joint training.
   - *iCaRL* introduces exemplar selection and prototype-based classification for incremental class learning.
2. **Regularization-Based Methods**
   - Techniques such as EWC penalize changes to important weights but rely on stationary task boundaries.
3. **Dynamic Architectures**
   - Progressive Networks and Dynamic Expansion methods grow model capacity, trading off memory and inference cost.
4. **Concept Drift Detection**
   - ADWIN offers adaptive windowing to detect statistically significant changes in streaming metrics.
5. **Open Source Tooling**
   - The River library and Avalanche provide streaming ML utilities and continual learning baselines, informing our implementation strategy.

---

## 3. Proposed Solution: Smart Rehearsal Pipeline
### 3.1 System Modules
- **Data Stream Ingestion**: Handles sequential tasks (Split CIFAR-10/100, Tiny-ImageNet) with configurable task orders.
- **Backbone Learner**: iCaRL-inspired encoder and classifier supporting exemplar buffers.
- **Monitoring Layer**: Records accuracy on exemplar-based validation streams.
- **ADWIN Trigger**: Evaluates metric streams to flag performance drops.
- **Adaptive Rehearsal Manager**: On drift detection, initiates intensive rehearsal (balanced fine-tuning, exemplar refresh).
- **Logging & Visualization**: Aggregates metrics (accuracy, forgetting, compute cost) into dashboards for analysis.

### 3.2 Data Flow
1. Receive new task batches.
2. Update the model using incremental training.
3. Evaluate held-out exemplar batches and feed results into ADWIN.
4. If drift detected → trigger rehearsal cycle; else continue streaming updates.
5. Periodically log metrics and update dashboards.

---

## 4. Methodology and Work Completed
| Phase | Timeline (Weeks) | Status | Key Artifacts |
|-------|------------------|--------|---------------|
| Literature Review & Baseline Plan | 1–2 | ✅ Completed | Survey notes, baseline experiment design |
| Baseline Reproduction (ER, GEM) | 2–3 | ✅ Completed | Baseline scripts, evaluation harness |
| iCaRL Implementation | 3–5 | 🔄 In Progress | Backbone architecture draft, exemplar policy module |
| ADWIN Integration & Calibration | 5–7 | 🔄 In Progress | Concept-drift monitor prototype, calibration notebook |
| Adaptive Rehearsal Prototype | 7–9 | ⏳ Scheduled | Integration plan, interface design |
| Extended Evaluation Setup | 9–11 | ⏳ Scheduled | Tiny-ImageNet stream, compute logging |
| Report & Presentation Prep | 12–15 | ⏳ Scheduled | Final report outline, deck skeleton |

---

## 5. Experimental Plan (for End-Semester Evaluation)
1. **Datasets**
   - Split CIFAR-10, Split CIFAR-100, Tiny-ImageNet (class-incremental protocols).
2. **Baselines**
   - Static Rehearsal (iCaRL with fixed replay schedule).
   - Experience Replay (fixed buffer, SGD).
   - Regularization Baselines (EWC, SI) as references if time allows.
3. **Metrics**
   - Accuracy per task and overall.
   - Forgetting measure (max accuracy drop per task).
   - Computational cost: number of replay steps, GPU time.
4. **Ablation Studies**
   - Vary ADWIN parameters (confidence, min window length).
   - Compare adaptive vs. fixed rehearsal frequency.
5. **Visualization**
   - Performance vs. time plots highlighting drift detections.
   - Compute vs. accuracy trade-off charts.

---

## 6. Risk Assessment & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| ADWIN sensitivity too high/low | False alarms or missed forgetting events | Parameter sweep, fallback to fixed schedule |
| Exemplar buffer overflow | Increased memory footprint | Adaptive pruning strategies, class-balanced selection |
| Limited compute resources | Longer training cycles | Early stopping strategies, smaller backbone for experiments |
| Integration complexity | Delayed prototype | Modular interfaces, incremental testing |

---

## 7. Deliverables for Mid-Semester Review
- Problem statement and motivation slides.
- Literature matrix comparing continual learning strategies.
- System architecture diagram of Smart Rehearsal pipeline.
- Baseline reproduction notebooks and logs.
- ADWIN calibration results on synthetic streams.
- Project timeline and risk register.

---

## 8. Roadmap to End-Semester Report
- Complete adaptive rehearsal integration and run full experiment suite.
- Generate comparative analysis (accuracy, forgetting, compute usage).
- Prepare reproducibility checklist and code documentation.
- Draft final report synthesizing quantitative and qualitative findings.
- Build final presentation deck and demo (if feasible).

---

## 9. References
1. Rebuffi, S.-A., et al. *"iCaRL: Incremental Classifier and Representation Learning."* CVPR, 2017.
2. van de Ven, G. M., and Tolias, A. S. *"Three Scenarios for Continual Learning."* arXiv:1904.07734, 2019.
3. Bifet, A., and Gavalda, R. *"Learning from Time-Changing Data with Adaptive Windowing."* SDM, 2007.
4. Montiel, J., et al. *"River: Machine Learning for Streaming Data in Python."* JMLR, 2021.
5. Lopez-Paz, D., and Ranzato, M. *"Gradient Episodic Memory for Continual Learning."* NIPS, 2017.
6. Kirkpatrick, J., et al. *"Overcoming Catastrophic Forgetting in Neural Networks."* PNAS, 2017.

---

## Appendix A. Presentation Script Outline
1. **Opening (1 min)**
   - Introduce continual learning challenge and stakes for real-world deployment.
   - State the Smart Rehearsal hypothesis.
2. **Motivation (2 min)**
   - Illustrate catastrophic forgetting and inefficiency of static rehearsal with visuals.
   - Position ADWIN as a proven tool for drift detection.
3. **Technical Approach (4 min)**
   - Walk through modular pipeline, emphasise monitoring-trigger loop.
   - Highlight how exemplar buffers and ADWIN interact.
4. **Progress to Date (2 min)**
   - Summarise completed baselines, prototype modules, and calibration experiments.
5. **Next Steps (2 min)**
   - Outline upcoming integration, evaluation, and documentation milestones.
6. **Closing (1 min)**
   - Reiterate contributions, expected benefits, and invite feedback.

## Appendix B. Resource Checklist
- **Compute**: Access to at least one GPU-equipped workstation (12GB+) for rehearsal experiments.
- **Software Stack**: PyTorch, Avalanche, River, Weights & Biases for experiment tracking.
- **Data Management**: Scripted downloaders for CIFAR variants and Tiny-ImageNet, with checksum verification.
- **Collaboration**: Shared Notion space for notes, GitHub Project board for task tracking.

---

## Appendix C. Risk Register Snapshot
| ID | Description | Probability | Impact | Owner | Mitigation |
|----|-------------|-------------|--------|-------|------------|
| R1 | ADWIN miscalibration causes oscillating triggers | Medium | High | Abhinav | Parameter tuning, smoothing metrics |
| R2 | Dataset download delays | Low | Medium | Abhinav | Pre-download, maintain mirror |
| R3 | GPU availability conflicts | Medium | Medium | Abhinav | Schedule runs off-peak, enable resume |
| R4 | Report scope creep | Medium | Low | Abhinav | Weekly check-ins, freeze feature list |

---

*Prepared for the Mid-Semester Review to showcase project vision, progress, and execution plan.*
