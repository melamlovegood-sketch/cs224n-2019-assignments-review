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

The assignment handout estimates 8-12 GPU hours for the full run. It was not rerun during this recovery session, so `outputs/test_outputs.txt` and a fresh full-run BLEU score are intentionally not fabricated. Run the full training and decoding commands in `run.sh` before a real Gradescope submission.

The written solution PDF is at `../../output/pdf/assignment5_written_solutions.pdf`.
