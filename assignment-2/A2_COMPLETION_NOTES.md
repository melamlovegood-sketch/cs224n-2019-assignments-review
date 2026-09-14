# Assignment 2 completion notes

## Completed work

- Implemented a numerically stable sigmoid.
- Implemented naive-softmax loss and gradients.
- Implemented negative-sampling loss and gradients, including repeated samples.
- Implemented skip-gram loss and gradient accumulation.
- Implemented stochastic gradient descent.
- Downloaded the Stanford Sentiment Treebank dataset.
- Trained the word vectors for 40,000 SGD iterations.
- Generated `word_vectors.png` and included it in the written solutions.

## Verification

Run from the `a2` Conda environment:

```bat
python word2vec.py
python sgd.py
python run.py
```

The word2vec numerical gradient checks pass, all SGD sanity checks pass, and the final exponentially smoothed training loss is approximately `9.812`, meeting the assignment's target of approximately 10 or below.

## Deliverables

- `assignment2.zip`: coding submission bundle.
- `a2_written_solutions.pdf`: written derivations and trained-vector plot.
- `a2_written_solutions.tex`: editable source for the written submission.
- `word_vectors.png`: visualization produced from the trained vectors.
