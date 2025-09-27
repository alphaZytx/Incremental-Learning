# Mid-Semester Presentation Blueprint

## 1. Why This Problem?
- **Continual Learning Challenge**: Sequential task training leads to catastrophic forgetting, undermining deployment in streaming environments.
- **Inefficiency of Static Rehearsal**: Fixed replay schedules consume compute even when the model is stable, motivating an adaptive alternative.
- **Guiding Hypothesis**: Coupling exemplar replay with drift-aware triggers can preserve accuracy while curbing redundant rehearsal cycles.

## 2. What Is the Proposed Solution?
- **Smart Rehearsal Pipeline**: Integrates an iCaRL-style backbone with the ADWIN concept-drift detector.
- **Modular Components**:
  1. Data stream ingestion for Split CIFAR-10/100 and Tiny-ImageNet.
  2. Exemplar-managed backbone that maintains class-balanced memories.
  3. Monitoring layer that logs validation metrics on exemplars.
  4. ADWIN trigger that flags statistically significant degradation.
  5. Adaptive rehearsal manager that launches focused replay bursts.
  6. Analytics hub that tracks forgetting, accuracy, and compute usage.
- **Key Differentiator**: Rehearsal intensity responds to drift rather than a static schedule, aligning compute with true performance needs.

## 3. How Will the Work Progress?
- **Phase 1 – Baselines**: Reproduce ER, GEM, and vanilla iCaRL to anchor comparisons.
- **Phase 2 – Monitoring**: Prototype ADWIN-based drift detection on synthetic and real validation streams.
- **Phase 3 – Integration**: Fuse monitoring signals with rehearsal scheduling and evaluate against static baselines.
- **Phase 4 – Evaluation**: Run comparative studies across datasets, ablations, and compute budgets.
- **Phase 5 – Reporting**: Package results into plots, tables, and narrative-ready summaries for end-sem submission.

## 4. When Are the Milestones?
| Week | Focus | Deliverables |
|------|-------|--------------|
| 1–2  | Literature survey, baseline planning | Annotated bibliography, experiment tracker |
| 3–4  | Baseline implementation | Verified scripts, sanity-check plots |
| 5–6  | ADWIN calibration | Drift detection notebook, threshold report |
| 7–8  | Adaptive loop prototype | Integrated training script, rehearsal logs |
| 9–11 | Extended evaluation | Cross-dataset metrics, forgetting analysis |
| 12–13| Optimisation & documentation | Performance tuning notes, reproducibility checklist |
| 14–15| Final report & presentation | End-semester report draft, slide deck, demo outline |

## 5. Which Insights Support the Narrative?
- **Empirical Evidence**: Baseline metrics vs. adaptive approach will show accuracy/forgetting trade-offs.
- **Literature Backbone**: iCaRL\cite{rebuffi2017icarl}, continual-learning surveys\cite{vandeven2019three}, ADWIN\cite{bifet2007adwin}, and River tooling\cite{montiel2021river} validate both methodology and tooling.
- **Storytelling Flow**: Introduce the challenge, compare strategy families, highlight inefficiencies, and land on Smart Rehearsal benefits backed by preliminary findings.

## 6. Presentation Toolkit
- **Architecture Diagram**: Layered view of ingestion, backbone, monitoring, trigger, and rehearsal manager.
- **Metric Dashboards**: Accuracy-over-time with drift markers, compute vs. accuracy scatter plots.
- **Risk Register**: Sensitivity analysis for ADWIN thresholds, memory constraints, compute availability.
- **Talking Points**: Prepared responses to "why adaptive?", "how robust?", and "what's next?" queries.

## 7. LaTeX Handout Template
```latex
\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{enumitem}
\usepackage{hyperref}

\title{Mid-Semester Presentation Handout\\Smart Rehearsal Project}
\author{Abhinav}
\date{September 2025}

\begin{document}
\maketitle

\section{Problem Motivation}
Catastrophic forgetting impairs continual learning deployments; static rehearsal wastes compute during stable phases. We hypothesise that drift-aware rehearsal sustains accuracy efficiently.\cite{rebuffi2017icarl,vandeven2019three}

\section{Pipeline Overview}
\begin{enumerate}[label=\textbf{Module \arabic*:}]
  \item Data stream ingestion (Split CIFAR-10/100, Tiny-ImageNet).
  \item iCaRL backbone with exemplar buffers.
  \item Metric monitor feeding ADWIN drift detector.\cite{bifet2007adwin}
  \item Adaptive rehearsal manager triggering replay bursts.
  \item Analytics hub logging accuracy, forgetting, compute cost.\cite{montiel2021river}
\end{enumerate}

\section{Progress Summary}
Baselines reproduced; monitoring prototype operational; integration roadmap validated with risk mitigations.

\section{Next Steps}
Complete adaptive loop experiments, extend to larger datasets, and craft end-semester report with comparative analysis.

\bibliographystyle{ieeetr}
\begin{thebibliography}{9}
\bibitem{rebuffi2017icarl}
S.-A. Rebuffi et al., ``iCaRL: Incremental Classifier and Representation Learning,'' \emph{CVPR}, 2017.

\bibitem{vandeven2019three}
G. M. van de Ven and A. S. Tolias, ``Three Scenarios for Continual Learning,'' arXiv:1904.07734, 2019.

\bibitem{bifet2007adwin}
A. Bifet and R. Gavald\'a, ``Learning from Time-Changing Data with Adaptive Windowing,'' \emph{SDM}, 2007.

\bibitem{montiel2021river}
J. Montiel et al., ``River: Machine Learning for Streaming Data in Python,'' \emph{JMLR}, 2021.
\end{thebibliography}

\end{document}
```

---

## 8. Quick-Reference Checklist
- [ ] Problem framing slide with key statistics on catastrophic forgetting.
- [ ] Literature slide contrasting rehearsal, regularisation, and dynamic architectures.
- [ ] Pipeline diagram with monitoring-trigger loop callouts.
- [ ] Baseline vs. adaptive rehearsal metric table.
- [ ] Risk mitigation slide and timeline snapshot.
- [ ] Appendix containing LaTeX handout and bibliography.

## 9. Presentation Q\&A Prep
| Question | Prepared Angle |
|----------|----------------|
| Why combine ADWIN with rehearsal? | Demonstrate compute savings and faster recovery from forgetting using detection-driven triggers. |
| How do you manage exemplar memory limits? | Describe class-balanced pruning and adaptive refresh strategies. |
| What if ADWIN misfires? | Outline parameter sweeps and fallback to fixed rehearsal schedules. |
| How scalable is the approach? | Discuss Tiny-ImageNet plans and modular integration for larger backbones. |
| What remains before end-semester? | Highlight integration testing, ablation studies, and report finalisation.

---

_Last updated: September 2025_
