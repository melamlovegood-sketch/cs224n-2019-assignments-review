# Assignment 3 completion notes

## Environment

- Python 3.10.20 (`cs224n-a1` Conda environment)
- PyTorch 2.14.0+cpu
- NumPy 1.26.4

## Implemented work

- `parser_transitions.py`: transition initialization, SHIFT, LEFT-ARC, RIGHT-ARC, full parse, and minibatch parsing.
- `parser_model.py`: trainable embeddings, hidden and output layers, Xavier initialization, embedding lookup, ReLU, dropout, and logits.
- `run.py`: Adam optimizer, cross-entropy loss, minibatch training, backpropagation, checkpoint loading, deterministic seeds, and full-data training.
- `a3_written_solutions.pdf`: written answers for Questions 1(a-b), 2(a-b), 2(e), and 2(f).

## Full training result

- Batch size: 1024
- Epochs: 10
- Learning rate: 0.0005
- Hidden size: 200
- Dropout probability: 0.5
- Best development UAS: 88.0250%
- Test UAS from the best development checkpoint: 88.6053%
- Saved checkpoint: `results/20260914_075525/model.weights`

## Verification

- `python parser_transitions.py part_c`: passed all SHIFT, LEFT-ARC, RIGHT-ARC, and parse checks.
- `python parser_transitions.py part_d`: passed the minibatch parser check.
- Python byte-compilation passed for the submission source files.
- Parser model forward smoke test passed with output shape `(8, 3)`.
- The written PDF compiled successfully and all three rendered pages were visually checked.

## Deliverables

- `assignment3.zip`: code and required data/utilities in the handout's requested archive layout.
- `a3_written_solutions.pdf`: written submission.
- `a3_written_solutions.tex`: editable report source.
