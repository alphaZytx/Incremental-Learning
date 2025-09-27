---
title: "Phase 3 Smart Rehearsal Playbook"
author: "Abhinav"
date: "September 2025"
---

# Phase 3: Smart Rehearsal Execution Guide

The Phase 3 milestone demonstrates how the Smart Rehearsal pipeline fuses the
static rehearsal backbone with adaptive drift monitoring and structured logging
for end-semester analysis. This playbook explains how to launch experiments,
interpret artefacts, and extend the workflow for comparative studies.

## 1. Objectives
- Validate the modular trainer that will feed the final report.
- Quantify light (interleaved mini-batches) versus heavy (full-buffer) rehearsal costs.
- Produce JSONL traces that can be aggregated into accuracy, forgetting, and compute tables.

## 2. Environment Checklist
1. Python 3.10+
2. `torch`, `torchvision`
3. Optional: `river` (for native ADWIN support). Without it, the fallback moving-average detector activates automatically.
4. Sufficient disk space for MNIST downloads (~50 MB).

## 3. Quick Start
```bash
python -m experiments.phase3.smart_rehearsal_pipeline \
  --epochs 3 \
  --buffer-size 400 \
  --full-rehearsal-epochs 2 \
  --detector-delta 0.02 \
  --output-dir outputs/phase3/run-001
```

### Key CLI Flags
| Flag | Purpose |
| --- | --- |
| `--dataset` | Currently `splitmnist`; extendable to future splits. |
| `--buffer-size` | Total exemplar capacity shared across classes. |
| `--full-rehearsal-epochs` | Number of heavy rehearsal passes triggered per drift. |
| `--detector-delta` | Sensitivity parameter for ADWIN / moving-average fallback. |
| `--eval-interval` | Frequency (in epochs) of detector updates. |

## 4. Output Structure
```
outputs/phase3/run-001/
  ├── events.jsonl    # per-epoch metrics + drift triggers
  └── summary.json    # aggregate rehearsal statistics
```

- **events.jsonl**: Append-only log of `TrainingStats` dictionaries. Load using
  pandas or the provided notebook template to compute forgetting curves.
- **summary.json**: Contains `light_rehearsal_steps` and `heavy_rehearsal_epochs`
  for quick efficiency comparisons.

## 5. Suggested Analyses
1. **Accuracy vs. Task Order** – Plot `accuracy` values filtered by `phase == "eval"` to track retention.
2. **Trigger Diagnostics** – Count rows with `detector_triggered == true` to examine drift sensitivity.
3. **Compute Budget** – Map heavy rehearsal epochs to wall-clock estimates to
   compare against fixed-schedule baselines from Phase 1.

## 6. Extending to Additional Datasets
- Introduce a new entry in `TASK_REGISTRY` with class ranges for Split CIFAR-10
  or CIFAR-100 once the backbone is swapped to a convolutional encoder.
- Replace `SimpleMLP` with a convolutional architecture and update the
  transform pipeline (normalisation, data augmentation) accordingly.

## 7. Reference Artefacts
- Phase 1 baseline script: `experiments/phase1/rehearsal_baseline.py`
- Phase 2 adaptive prototype: `experiments/phase2/adaptive_rehearsal.py`
- End-semester coding plan: `docs/end-semester-coding-plan.md`

These materials, together with this playbook, cover the story arc required for
mid-semester demonstrations and position the project for the final evaluation.

---

*Prepared for review with faculty mentors and industry evaluators.*
