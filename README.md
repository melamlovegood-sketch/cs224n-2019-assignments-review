# CS224n 2019 — Assignment and Final Project Portfolio

This private repository collects my completed and recovered coursework for Stanford CS224n (Winter 2019), including assignments 1–5 and the default final project. It is organized as a concise mentor-review snapshot: source code, written solutions, selected outputs, verification notes, and compact submission archives are included; large model checkpoints, duplicated datasets, caches, and reference-solution directories are intentionally excluded.

## Assignment overview

| Assignment | Topic | Evidence included | Verified result |
|---|---|---|---|
| [A1](assignment-1/) | Exploring word vectors | Completed notebook, exported PDF, helper implementation | Notebook deliverable preserved |
| [A2](assignment-2/) | Word2Vec derivations and implementation | Source, written PDF, trained-vector plot, submission ZIP | Gradient/SGD checks passed; final smoothed loss about 9.812 |
| [A3](assignment-3/) | Dependency parsing | Source, written PDF, submission ZIP, completion notes | Dev UAS 88.0250%; test UAS 88.6053% |
| [A4](assignment-4/) | Neural machine translation with RNNs and attention | Source, written PDF, decoded output, submission ZIP | Official checks 1d/1e/1f passed; corpus BLEU 21.550124 |
| [A5](assignment-5/) | Character-aware NMT | Source, written PDF, full decoded output, submission ZIP, recovery notes | Official checks passed; full-corpus BLEU 24.4811 |
| [Final project](final-project/) | Reliable extractive QA on SQuAD 2.0 | Reproducible code, raw results, proposal, milestone, final report, poster, submission ZIP | Held-out EM 62.53, F1 64.67; no-answer accuracy 70.94% |

## Repository layout

Each assignment folder contains the readable implementation and any written/report artifacts. Files named `assignmentN.zip` are the compact submission bundles produced for that assignment. The final-project folder additionally contains machine-readable metrics, predictions, figures, qualitative error analysis, and the complete written deliverables.

The Markdown completion/recovery notes document the environment, checks, and measured results. A5's note records the completed full-corpus GPU run, checkpoint recovery after a power interruption, and fresh local BLEU score. The final-project README records its disjoint calibration/evaluation protocol and explicitly distinguishes the public-subset result from the original private course leaderboard.

## Reproducibility notes

- The coursework originally targeted older Python and PyTorch versions; compatibility adjustments are documented in the source and notes.
- Large trained checkpoints and optimizer states are kept locally and excluded from Git history.
- Large public course datasets are present only when required inside an existing submission archive, not duplicated as browsable files.
- This repository is intended for private academic review and is not an official Stanford repository.
