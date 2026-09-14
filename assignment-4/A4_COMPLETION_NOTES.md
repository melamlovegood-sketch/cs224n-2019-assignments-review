# CS224n Assignment 4 completion notes

## Environment

- Windows 11
- Conda environment: `d2l`
- Python 3.9.25
- PyTorch 2.8.0+cu129
- NVIDIA RTX 5060 Laptop GPU (CUDA enabled)

## Implemented work

- `utils.py`: sentence padding
- `model_embeddings.py`: source and target embeddings
- `nmt_model.py`: bidirectional encoder, decoder, attention step, masks, beam search compatibility, model loading
- `run.py`: current PyTorch compatibility, CUDA performance settings, checkpoint resume support
- `sanity_check.py`: current PyTorch compatibility

## Verification

- Official sanity checks `1d`, `1e`, and `1f`: passed
- Python compilation check: passed
- One-batch forward/backward/optimizer smoke test: passed
- Full local GPU training completed
- Best development perplexity: 17.483624
- Full 8,064-sentence test decoding completed
- Corpus BLEU: 21.550124 (assignment threshold: greater than 21)

## Main artifacts

- `model.bin`: best locally trained checkpoint
- `outputs/test_outputs.txt`: translations produced by that checkpoint
- `outputs/BLEU.txt`: corpus BLEU record
- `a4_written_solutions.pdf`: written answers
- `assignment4.zip`: code/data/output submission archive

The written answers and error-analysis examples were updated to match the final locally trained output.
