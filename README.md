# CS224n 2019 — Assignment Portfolio

This private repository collects my completed and recovered coursework for Stanford CS224n (Winter 2019). It is organized as a concise mentor-review snapshot: source code, written solutions, selected outputs, verification notes, and the original submission archives are included; large model checkpoints, duplicated datasets, caches, and reference-solution directories are intentionally excluded.

## Assignment overview

| Assignment | Topic | Evidence included | Verified result |
|---|---|---|---|
| [A1](assignment-1/) | Exploring word vectors | Completed notebook, exported PDF, helper implementation | Notebook deliverable preserved |
| [A2](assignment-2/) | Word2Vec derivations and implementation | Source, written PDF, trained-vector plot, submission ZIP | Gradient/SGD checks passed; final smoothed loss about 9.812 |
| [A3](assignment-3/) | Dependency parsing | Source, written PDF, submission ZIP, completion notes | Dev UAS 88.0250%; test UAS 88.6053% |
| [A4](assignment-4/) | Neural machine translation with RNNs and attention | Source, written PDF, decoded output, submission ZIP | Official checks 1d/1e/1f passed; corpus BLEU 21.550124 |
| [A5](assignment-5/) | Character-aware NMT | Source, written PDF, focused outputs, submission ZIP, recovery notes | Official component sanity checks passed; focused tests 4/4 |

## Repository layout

Each assignment folder contains the readable implementation and any written/report artifacts. Files named `assignmentN.zip` are the compact submission bundles produced for that assignment.

The Markdown completion/recovery notes document the environment, checks, and measured results. A5's note also records that the time-intensive full-corpus retraining was not repeated during recovery; no fresh full-run BLEU score is claimed.

## Reproducibility notes

- The coursework originally targeted older Python and PyTorch versions; compatibility adjustments are documented in the source and notes.
- Large trained checkpoints and optimizer states are kept locally and excluded from Git history.
- Large public course datasets are present only when required inside an existing submission archive, not duplicated as browsable files.
- This repository is intended for private academic review and is not an official Stanford repository.

