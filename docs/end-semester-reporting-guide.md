# End-Semester Reporting Guide

This "mini-README" walks through the exact commands, artefacts, and LaTeX snippets required to translate the Smart Rehearsal experiments into a polished final report. Use it as the checklist when regenerating results or mentoring teammates on how to present the project.

---

## 1. Environment Preparation
1. **Install Core Dependencies** (adjust versions as needed):
   ```bash
   pip install torch torchvision torchaudio
   pip install matplotlib seaborn
   pip install river
   ```
2. **Clone the Repository and Navigate to It**:
   ```bash
   git clone <repo-url>
   cd Incremental-Learning
   ```
3. **(Optional) Configure Output Directory**: Set `OUTPUT_DIR=outputs` if you wish to centralise artefacts.

> **Tip:** Use virtual environments (`python -m venv .venv && source .venv/bin/activate`) to isolate dependencies for reproducibility.

---

## 2. Run the Experiments
Execute each phase with consistent hyper-parameters so comparisons remain fair. Increase `--epochs` or buffer sizes for the final run once the pipeline is stable.

### 2.1 Phase 1 — Static Rehearsal Baseline
```bash
python -m experiments.phase1.rehearsal_baseline \
  --dataset split_mnist \
  --epochs 5 \
  --buffer-size 200 \
  --export-dir outputs/phase1 \
  --plot
```
**Collect:**
- `outputs/phase1/metrics.csv`
- `outputs/phase1/rehearsal_stats.json`
- `outputs/phase1/phase1_accuracy.png`

### 2.2 Phase 2 — Adaptive Rehearsal Prototype
```bash
python -m experiments.phase2.adaptive_rehearsal \
  --dataset split_mnist \
  --epochs 5 \
  --buffer-size 200 \
  --detector-delta 0.0025 \
  --export-dir outputs/phase2 \
  --plot
```
**Collect:**
- `outputs/phase2/metrics.json`
- `outputs/phase2/drift_events.jsonl`
- `outputs/phase2/phase2_accuracy.png`

### 2.3 Phase 3 — Smart Rehearsal Pipeline
```bash
python -m experiments.phase3.smart_rehearsal_pipeline \
  --dataset split_mnist \
  --epochs 5 \
  --buffer-size 200 \
  --detector-delta 0.0025 \
  --evaluation-interval 50 \
  --export-dir outputs/phase3
```
**Collect:**
- `outputs/phase3/events.jsonl`
- `outputs/phase3/summary.json`
- `outputs/phase3/buffer_usage.csv`

### 2.4 Aggregated Comparison
```bash
python -m experiments.end_semester.comparison_suite \
  --phase1-dir outputs/phase1 \
  --phase2-dir outputs/phase2 \
  --phase3-dir outputs/phase3 \
  --export-dir outputs/end_semester
```
**Collect:**
- `outputs/end_semester/comparison_report.json`
- `outputs/end_semester/comparison_report.md`
- Terminal summary (copy for the appendix).

---

## 3. Building the Report Narrative
### 3.1 Suggested Structure
1. **Title & Abstract** – Use the executive summary in `docs/end-semester-report.md`.
2. **Introduction** – Motivate catastrophic forgetting and static rehearsal inefficiency.
3. **Methodology** – Summarise pipeline components with references.
4. **Results** – Insert tables/figures described below.
5. **Discussion** – Interpret adaptive gains and risk mitigations.
6. **Conclusion & Future Work** – Reuse Section 7 of the report template.

### 3.2 Tables and Figures
| Artefact | How to Generate | How to Include |
| --- | --- | --- |
| Comparative metrics table | `comparison_report.md` | Copy into LaTeX via `[markdown]` package or convert to `tabular` using Pandoc. |
| Accuracy plots | Generated via `--plot` flags in Phases 1-2 | Include as `\includegraphics` in the Evaluation section. |
| Drift event timeline | Summarise events from `drift_events.jsonl` / `events.jsonl` | Create a timeline figure or table highlighting detection timestamps. |
| Buffer utilisation chart | Derive from `buffer_usage.csv` (Phase 3) | Plot using matplotlib; embed as figure showing efficiency. |

### 3.3 Comparative Commentary Prompts
- **Accuracy**: Highlight parity or gains vs. baseline.
- **Efficiency**: Quantify reduction in rehearsal steps or runtime.
- **Robustness**: Discuss drift detector sensitivity and fallback behaviour.
- **Scalability**: Mention dataset extensions and modular design.

---

## 4. LaTeX Snippets
### 4.1 Importing Markdown Table
```latex
\begin{table}[t]
  \centering
  \input{comparison_report.tex} % if converted via pandoc
  \caption{Comparison of static vs. adaptive rehearsal pipelines on SplitMNIST.}
  \label{tab:comparison}
\end{table}
```
Run `pandoc outputs/end_semester/comparison_report.md -o comparison_report.tex` to convert.

### 4.2 Referencing Figures
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.75\linewidth]{figures/phase2_accuracy.png}
  \caption{Accuracy trajectory with ADWIN-triggered rehearsal bursts.}
  \label{fig:phase2_accuracy}
\end{figure}
```

### 4.3 Appendix Checklist
```latex
\section*{Reproducibility Checklist}
\begin{itemize}
  \item Git commit hash: \texttt{<hash>}
  \item Commands executed: see `docs/end-semester-reporting-guide.md`
  \item Hardware: NVIDIA RTX 3080 (12 GB), 32 GB RAM
  \item Software: Python 3.10, PyTorch 2.2, River 0.19
\end{itemize}
```

---

## 5. Presentation Tips
- Lead with the problem statement before revealing the adaptive twist.
- Use the aggregated Markdown table as a slide to anchor discussion.
- Showcase at least one drift event timeline to emphasise responsiveness.
- Close with a roadmap referencing future research directions (detector variants, task-free CL).

---

## 6. Submission Checklist
- [ ] All experiments executed with consistent seeds/hyper-parameters.
- [ ] Outputs copied to `outputs/end_semester/` and backed up.
- [ ] `comparison_report.md` converted to LaTeX or pasted as a table.
- [ ] Figures exported to `figures/` and referenced in the report.
- [ ] `docs/end-semester-report.md` reviewed and tailored to actual results.
- [ ] README updated with links to this guide and final report.

> **Final Reminder:** Tag artefacts with dataset, buffer size, and detector parameters so reviewers can trace every number in the comparative analysis.
