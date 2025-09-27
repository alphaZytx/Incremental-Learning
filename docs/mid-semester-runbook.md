# Mid-Semester Experiment Runbook

This guide explains how to execute the Phase 1 rehearsal baseline, collect metrics for the mid-semester review, and incorporate comparative visuals into the LaTeX report.

## 1. Environment Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
2. **Install dependencies:**
   ```bash
   pip install torch torchvision matplotlib
   ```
   *For GPU acceleration, install the CUDA-enabled wheel from the [official PyTorch instructions](https://pytorch.org/get-started/locally/).*  
   Additional tools such as `jupyter` or `tensorboard` can be installed on demand.

## 2. Run the Phase 1 Baseline

The first milestone is to reproduce a rehearsal baseline on SplitMNIST. This serves as the comparison point for the adaptive Smart Rehearsal pipeline.

```bash
python -m experiments.phase1.rehearsal_baseline \
    --epochs 5 \
    --buffer-size 200 \
    --plot
```

### Command Arguments
- `--epochs`: Training epochs per task (default: 5).
- `--buffer-size`: Total capacity of the exemplar buffer (default: 200).
- `--batch-size`: Mini-batch size for SGD (default: 64).
- `--lr`: Learning rate (default: 1e-3).
- `--output-dir`: Where metrics and plots are stored (default: `outputs/phase1`).
- `--plot`: Save `phase1_accuracy.png` for presentation/report use.

After the run completes, the following artefacts are produced:

| File | Description |
| --- | --- |
| `outputs/phase1/phase1_metrics.json` | Per-task accuracy, seen classes, buffer usage. |
| `outputs/phase1/phase1_metrics.csv` | CSV view of the same metrics for spreadsheets. |
| `outputs/phase1/phase1_accuracy.png` | Accuracy curve (only when `--plot` is enabled). |

## 3. Run the Phase 2 Adaptive Prototype

Once the baseline numbers are locked, execute the concept-drift-aware prototype to demonstrate progress toward the Smart Rehearsal vision.

```bash
python -m experiments.phase2.adaptive_rehearsal \
    --epochs 3 \
    --buffer-size 300 \
    --full-rehearsal-epochs 2 \
    --plot
```

### Key Flags
- `--detector-delta`, `--detector-min-window`, `--detector-fallback-drop`: Tune ADWIN sensitivity (or the fallback heuristic when River is unavailable).
- `--full-rehearsal-epochs`: Controls how intensive each triggered rehearsal burst becomes.
- `--light-rehearsal-ratio`: Probability of mixing buffer exemplars during standard SGD updates.

### Output Artefacts

| File | Description |
| --- | --- |
| `outputs/phase2/phase2_summary.json` | Combined metrics and drift events for scripted analysis. |
| `outputs/phase2/phase2_metrics.csv` | Task-level accuracy snapshot mirroring the baseline output for easy comparison. |
| `outputs/phase2/phase2_events.csv` | Timestamped drift triggers and the number of rehearsal steps executed. |
| `outputs/phase2/phase2_accuracy.png` | Accuracy curve annotated with detected drifts (requires `--plot`). |

Use these results to illustrate how adaptive rehearsal differs from the static baseline in the mid-semester conversation—e.g., highlight fewer rehearsal bursts while retaining accuracy.

## 4. Comparative Analysis Checklist

To prepare mid-semester slides or documentation:

1. **Baseline Table:** Import `phase1_metrics.csv` into Google Sheets/Excel to summarise accuracy after each incremental task.
2. **Buffer Utilisation:** Highlight the `rehearsal_examples` column to show memory footprint.
3. **Narrative Hooks:** Connect accuracy plateau/drops to hypotheses about static rehearsal.

These artefacts become the "before" scenario for the end-semester adaptive experiments.

## 5. Embedding Results in LaTeX

1. Copy `outputs/phase1/phase1_accuracy.png` into your report's `figures/` directory.
2. Reference it in LaTeX:
   ```latex
   \begin{figure}[h]
     \centering
     \includegraphics[width=0.7\textwidth]{figures/phase1_accuracy.png}
     \caption{Phase 1 rehearsal baseline accuracy across SplitMNIST tasks.}
     \label{fig:phase1-baseline}
   \end{figure}
   ```
3. For tabular metrics, convert the CSV into a LaTeX table using an online converter or the `pandas.DataFrame.to_latex()` method.
4. Cite relevant literature inline (e.g., \cite{rebuffi2017icarl}) to contextualise the baseline.

## 6. Next Steps (Preview)

- **Phase 2:** Integrate ADWIN monitoring to detect drift triggers.
- **Phase 3:** Implement adaptive rehearsal policy and rerun experiments.
- **Phase 4:** Prepare comparative plots (static vs. adaptive) and discuss compute savings.

Document results from each phase in the same `outputs/` hierarchy to maintain a reproducible audit trail.
