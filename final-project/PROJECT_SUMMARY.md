# Project completion summary

**Title:** Calibrated DistilBERT for Reliable Extractive Question Answering

**Route:** CS224N default final project, pretrained contextual embeddings (PCE)

## Final held-out results

| System | EM | F1 | Answerable F1 | Unanswerable accuracy |
|---|---:|---:|---:|---:|
| SQuAD 1.1 checkpoint, threshold 0 | 38.00 | 41.79 | 84.37 | 0.00 |
| SQuAD 2.0 fine-tuned, threshold 0 | 61.27 | 63.92 | 67.53 | 60.37 |
| SQuAD 2.0 fine-tuned + calibration | **62.53** | **64.67** | 58.29 | **70.94** |

The final evaluation set contains 1,500 examples (743 answerable and 757
unanswerable). The no-answer threshold was selected on a separate 1,500-example
calibration split. Training used 12,000 examples for two epochs and took 274
seconds on an RTX 5060 Laptop GPU.

## Deliverables

- `deliverables/output/project_proposal.pdf`
- `deliverables/output/project_milestone.pdf`
- `deliverables/output/final_report.pdf`
- `deliverables/output/project_poster.pdf`
- `deliverables/output/project_code.zip`
- `deliverables/team_contributions.txt`
- `project/model/` trained checkpoint and tokenizer
- `project/results/` metrics, predictions, figures, and error analysis
