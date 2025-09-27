# End-Semester Verification Checklist

This document confirms the current readiness of the Smart Rehearsal project before finalising the end-semester submission. Each section points to the artefacts that demonstrate the work completed so far and highlights any remaining follow-ups.

## 1. Project Idea & Narrative
- **Problem framing** – The README summarises catastrophic forgetting, motivates adaptive rehearsal, and connects the roadmap to the mid- and end-semester artefacts so reviewers can trace the storyline quickly.【F:README.md†L1-L93】
- **Research grounding** – The mid-semester dossier documents the literature review, system design, and planned experiments, providing the context needed for evaluation discussions.【F:docs/mid-semester-review.md†L1-L172】
- **Implementation roadmap** – The end-semester coding plan captures module responsibilities, risk mitigations, and milestone sequencing to show how the idea evolves into deliverables.【F:docs/end-semester-coding-plan.md†L1-L95】

## 2. Code Readiness Across Phases
- **Phase 1 baseline** – `experiments/phase1/rehearsal_baseline.py` trains the SplitMNIST rehearsal baseline with exemplar buffering, deterministic seeding, metrics export, and CLI configuration, making it suitable for comparative plots in the report.【F:experiments/phase1/rehearsal_baseline.py†L1-L153】
- **Phase 2 adaptive rehearsal** – `experiments/phase2/adaptive_rehearsal.py` layers ADWIN-based drift detection onto the baseline, emitting JSONL logs that capture rehearsal decisions and detector responses for mid vs. end-semester comparison tables.【F:experiments/phase2/adaptive_rehearsal.py†L1-L160】
- **Phase 3 smart pipeline** – `experiments/phase3/smart_rehearsal_pipeline.py` unifies modular components (data streaming, detector calibration, rehearsal management, logging) to produce the final adaptive workflow showcased in the end-semester presentation.【F:experiments/phase3/smart_rehearsal_pipeline.py†L1-L163】
- **Comparison tooling** – `experiments/end_semester/comparison_suite.py` aggregates phase metrics, computes forgetting proxies, and renders Markdown/JSON summaries so results can be dropped straight into the report.【F:experiments/end_semester/comparison_suite.py†L1-L132】

## 3. Reporting Artefacts
- **End-semester report skeleton** – The template in `docs/end-semester-report.md` outlines executive summary, methodology, results, and reflection sections, ensuring consistent structure for the final document.【F:docs/end-semester-report.md†L1-L128】
- **Evaluation & reporting guides** – Step-by-step instructions for running experiments, capturing artefacts, and embedding outputs into LaTeX are documented in `docs/end-semester-evaluation-guide.md` and `docs/end-semester-reporting-guide.md`, covering both execution flow and storytelling tips.【F:docs/end-semester-evaluation-guide.md†L1-L106】【F:docs/end-semester-reporting-guide.md†L1-L163】
- **Phase-specific playbooks** – The mid-semester runbook and Phase 3 playbook provide walk-throughs for baseline reproduction and smart pipeline demos so supporting evidence is straightforward to generate.【F:docs/mid-semester-runbook.md†L1-L105】【F:docs/phase3-smart-playbook.md†L1-L78】

## 4. Verification Status
- ✅ `python -m compileall experiments` confirms that all experiment modules are syntactically valid in this environment. Run full training only on a machine with PyTorch/River installed to reproduce metrics.【276fbb†L1-L12】
- ⚠️ Runtime dependencies (`torch`, `torchvision`, `river`) remain external; ensure the environment setup instructions in the runbooks are followed before executing training scripts.

## 5. Next Steps Before Final Submission
1. Execute each phase on the target hardware to produce fresh metrics and plots (export to `outputs/` as described in the guides).
2. Feed the artefacts into the comparison suite to generate consolidated tables and Markdown snippets for the end-semester report.
3. Update the final report (`docs/end-semester-report.md`) with the new quantitative results, qualitative analysis, and visual assets.
4. Prepare presentation slides mirroring the structure from the README and reporting guides so the narrative remains consistent across deliverables.

Once these steps are complete, the project will have end-to-end evidence for the adaptive rehearsal strategy, making it ready for professor review.
