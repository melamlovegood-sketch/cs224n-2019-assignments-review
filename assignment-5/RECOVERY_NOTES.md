# Assignment 5 recovery notes

## Completed

- Character indexing, word/character padding, and character tensor conversion.
- Character CNN, Highway layer, and character-based model embeddings.
- Assignment 4 encoder/decoder/attention integration using character inputs.
- Character LSTM decoder initialization, forward pass, summed training loss, and batched greedy decoding.
- Compatibility fixes for current PyTorch releases.
- Generated `vocab.json`, `vocab_tiny_q1.json`, and `vocab_tiny_q2.json`.

## Verification

- Official sanity checks passed: `1e`, `1f`, `1j`, `2a`, `2b`, `2c`, `2d`.
- `python test_components.py`: 4/4 focused tests passed.
- End-to-end NMT smoke test: finite forward score, finite backward gradients, and successful beam search.
- `outputs/test_outputs_local_q1.txt`: Corpus BLEU `99.29792465574434`.
- `outputs/test_outputs_local_q2.txt`: Corpus BLEU `99.29792465574434`.

## Long full-corpus run

The full English-Spanish model was trained locally on an NVIDIA GPU. A power loss interrupted the original process during epoch 21, so training resumed from the latest saved model and optimizer checkpoint. The configured early-stopping criterion ended training at iteration 196,000.

- Best development perplexity: `342.893788`.
- `outputs/test_outputs.txt`: all 8,064 test sentences decoded.
- Fresh local corpus BLEU: `24.48109275208612`, above the 22.5 full-credit threshold.

The final written solution is `assignment5_written_solutions.pdf`.
