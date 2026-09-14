# Final project — Calibrated DistilBERT for reliable extractive QA

This directory contains the completed CS224N default final project. It studies
whether SQuAD 2.0 fine-tuning and held-out no-answer threshold calibration can
improve a compact extractive reader originally trained on SQuAD 1.1.

## Final held-out results

| System | EM | F1 | Answerable F1 | Unanswerable accuracy |
|---|---:|---:|---:|---:|
| SQuAD 1.1 checkpoint, threshold 0 | 38.00 | 41.79 | 84.37 | 0.00 |
| SQuAD 2.0 fine-tuned, threshold 0 | 61.27 | 63.92 | 67.53 | 60.37 |
| SQuAD 2.0 fine-tuned + calibration | **62.53** | **64.67** | 58.29 | **70.94** |

The comparison uses 12,000 training examples, 1,500 calibration examples, and
a disjoint 1,500-example evaluation set selected with seed 224. The reported
numbers are an internal evaluation on a public SQuAD 2.0 validation subset, not
the original private CS224N leaderboard.

## Run

Install the dependencies, run the lightweight checks, and then start either a
smoke experiment or the full configured experiment:

```powershell
python -m pip install -r requirements.txt
python test_experiment.py
python experiment.py --smoke
python experiment.py
```

The experiment downloads the public dataset and starting checkpoint when they
are not already cached. A CUDA GPU is recommended for the full run.

## Contents

- `experiment.py`: preprocessing, training, decoding, calibration, evaluation,
  plotting, and error analysis.
- `test_experiment.py`: fast metric and threshold correctness checks.
- `results/`: raw metrics, predictions, training history, figures, and selected
  qualitative cases from the completed run.
- `deliverables/`: proposal, milestone, eight-page final report, poster, and the
  compact course code-submission ZIP.
- `report-source/`: reproducible ReportLab builder, bibliography, and contribution
  statement used for the PDF deliverables.

The trained checkpoint and downloaded dataset are intentionally excluded from
Git history. Exact package versions, hardware details, split sizes, timing, and
hyperparameters are recorded in `results/results.json`.
