---
title: "End-Semester Implementation Plan"
author: "Abhinav"
date: "September 2025"
---

# Smart Rehearsal Coding Plan for End-Semester Evaluation

## 1. Goal Alignment
- **Objective**: Deliver a production-ready prototype that fuses static rehearsal strength (iCaRL) with adaptive monitoring (ADWIN) to surpass baseline accuracy-for-cost trade-offs.
- **Key Question**: How does the combined strategy outperform pure rehearsal or pure drift detection in retaining past knowledge while scaling to new tasks?
- **Success Criteria**: Demonstrate superior average accuracy, reduced forgetting, and lower cumulative replay budget versus fixed-schedule baselines across class-incremental benchmarks.

## 2. Current Assets Recap
| Asset | Status | Notes |
|-------|--------|-------|
| Literature review + baseline scripts | ✅ Completed | ER, GEM reference implementations ready for comparison |
| iCaRL backbone draft | 🔄 Partial | Needs refactor into modular trainer/service layers |
| ADWIN monitor prototype | 🔄 Partial | Works on synthetic drift; pending integration hooks |
| Logging notebooks | ✅ Available | To be promoted into reusable utilities |

## 3. Target Architecture Overview
```
Streaming Dataset  -->  Data Loader  -->  Incremental Trainer  -->  Metrics Buffer  -->  ADWIN Monitor
                                                   |                                    |
                                                   v                                    |
                                      Exemplar Memory Manager <-------------------------
                                                   |
                                                   v
                                           Adaptive Rehearsal Scheduler
```
- **Incremental Trainer**: Maintains feature extractor + nearest-mean classifier heads.
- **Metrics Buffer**: Collects rolling validation accuracy/loss for drift analysis.
- **Adaptive Scheduler**: Chooses between light-touch rehearsal (mini replay) and heavy refresh (balanced fine-tuning + exemplar update).

## 4. Module Breakdown & Coding Tasks
### 4.1 Data & Experiment Orchestration
- Implement a unified CLI (`train.py`) that configures dataset splits, buffer sizes, and ADWIN parameters.
- Add reproducible experiment manifests (YAML) to describe runs for CIFAR-10/100 and Tiny-ImageNet.

### 4.2 Backbone Trainer Enhancements
- Refactor iCaRL code into: `FeatureExtractor`, `ExemplarManager`, and `IncrementalLearner` classes.
- Introduce hooks for lightweight rehearsal steps (mini-batch replay) and full balanced fine-tuning.
- Ensure exemplar selection uses herding with class-balanced quotas.

### 4.3 Monitoring & Adaptive Control
- Wrap River's ADWIN in a `DriftDetector` interface with reset, confidence, and window diagnostics.
- Design a `MetricStream` publisher that feeds accuracy/loss events to detectors asynchronously.
- Implement policy logic: `if detector.triggered(): schedule_full_rehearsal()` else continue light replay.

### 4.4 Evaluation & Comparative Analysis
- Create evaluation scripts to compute per-task accuracy, forgetting, and cumulative replay counts.
- Add plotting utilities for accuracy vs. time and compute vs. accuracy trade-offs.
- Automate baselines (static rehearsal, ER) with identical experiment manifests for fair comparison.

### 4.5 Engineering Quality
- Integrate Hydra or argparse-based configuration logging for reproducibility.
- Add unit tests for exemplar selection, detector triggering, and rehearsal scheduling decisions.
- Configure CI hooks (pytest + lint) to run on push.

## 5. Development Iterations
| Iteration | Focus | Deliverables |
|-----------|-------|--------------|
| I1 (Week 8-9) | Modularize trainer & memory | Refactored classes, baseline parity tests |
| I2 (Week 9-10) | Integrate ADWIN control loop | Drift detector interface, event logging |
| I3 (Week 10-11) | Adaptive rehearsal policies | Policy unit tests, initial ablation results |
| I4 (Week 11-12) | Benchmark automation | Manifest-driven runs, reproducibility scripts |
| I5 (Week 12-13) | Analysis & visualization | Comparison plots, trade-off tables |
| I6 (Week 13-14) | Polish & documentation | API docs, deployment checklist |

## 6. Comparative Advantage Narrative
- **Static Rehearsal vs. Smart Rehearsal**: Expect lower replay volume (compute) while matching or exceeding accuracy due to targeted interventions.
- **Pure Drift Detection vs. Smart Rehearsal**: ADWIN alone detects change but cannot recover performance; coupling with rehearsal yields measurable retention gains.
- **Combined Storyline**: Present joint metrics (accuracy, forgetting, replay steps) demonstrating Smart Rehearsal dominates both baselines on efficiency-frontier plots.

## 7. Risk Mitigation & Contingencies
| Risk | Mitigation |
|------|------------|
| ADWIN false positives causing excess rehearsal | Parameter sweep, incorporate patience counter before triggering full refresh |
| Memory growth with many classes | Implement adaptive pruning + reservoir sampling fallback |
| Training instability on Tiny-ImageNet | Use mixed-precision and gradient clipping; provide smaller backbone alternative |
| Time constraints for full ablations | Prioritize CIFAR-100 results; treat Tiny-ImageNet as stretch goal |

## 8. Documentation & Presentation Artifacts
- Maintain a changelog highlighting integration milestones and experimental outcomes.
- Prepare slide-ready figures: system architecture, comparison plots, replay budget chart.
- Draft narrative linking literature gap → implementation → empirical evidence for the end-semester defense.

## 9. Reference Backbone
1. Rebuffi, S.-A., et al. "iCaRL: Incremental Classifier and Representation Learning." CVPR 2017.
2. van de Ven, G. M., and Tolias, A. S. "Three Scenarios for Continual Learning." arXiv:1904.07734, 2019.
3. Bifet, A., and Gavaldà, R. "Learning from Time-Changing Data with Adaptive Windowing." SDM 2007.
4. Hayes, T. L., et al. "REMIND Your Neural Network to Prevent Catastrophic Forgetting." ECCV 2020.
5. Montiel, J., et al. "River: Machine Learning for Streaming Data in Python." JMLR 2021.

