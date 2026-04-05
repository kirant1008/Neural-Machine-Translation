# Neural Machine Translation (English to Vietnamese)

A seq2seq neural machine translation system built with PyTorch, translating English to Vietnamese using a bidirectional LSTM encoder-decoder architecture trained on the IWSLT15 dataset.

## Architecture

- **Encoder**: Bidirectional LSTM (hidden size 1024, 1 layer)
- **Decoder**: Unidirectional LSTM with teacher forcing during training
- **Training**: Adam optimizer (lr=0.0001), CrossEntropyLoss, batch size 64, 40 epochs
- **Evaluation**: BLEU-1 score via NLTK

## Dataset

[IWSLT15 English-Vietnamese](https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/) from Stanford NLP (~19.4k training, ~500 test sentences). Sentences are filtered to a max length of 25 tokens.

## Setup

Requires Python 3.10+ and a CUDA-capable GPU.

```bash
uv sync
```

## Usage

**Train** the model:

```bash
uv run python NMT.py train
```

Model weights are saved to `./model/mdl_weights.pth`.

**Test** the model (computes BLEU score and shows sample translations):

```bash
uv run python NMT.py test
```

**Translate** interactively:

```bash
uv run python NMT.py translate
```

## Project Structure

| File | Description |
|---|---|
| `NMT.py` | Main entry point — data loading, training, testing, and translation |
| `nmt_model.py` | Model architecture (encoder, decoder, seq2seq) |
| `preprocess.py` | Vocabulary, tokenization, tensor conversion, batching |

## References

1. https://tsdaemon.github.io/2018/07/08/nmt-with-pytorch-encoder-decoder.html
2. https://medium.com/dair-ai/neural-machine-translation-with-attention-using-pytorch-a66523f1669f
