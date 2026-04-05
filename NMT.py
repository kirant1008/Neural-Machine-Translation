"""Neural Machine Translation — English to Vietnamese using seq2seq with LSTM."""

import argparse
import os

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.optim import Adam
from tqdm import tqdm

import nmt_model as nmt
import preprocess as pp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_LANG = "en"
TARGET_LANG = "vi"
MAX_LEN = 25
MIN_WORD_COUNT = 1
HIDDEN_SIZE = 1024
N_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-4
GRAD_CLIP = 5.0
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "mdl_weights.pth")
SEED = 42

TOKENIZERS = {
    "en": nltk.tokenize.WordPunctTokenizer().tokenize,
    "vi": nltk.tokenize.WordPunctTokenizer().tokenize,
}


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available, using CPU. Training will be very slow.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_parallel_data(url_src: str, url_tgt: str) -> pd.DataFrame:
    """Load parallel text from URLs into a DataFrame."""
    text_src = pd.read_csv(url_src, sep="\n", header=None)
    text_tgt = pd.read_csv(url_tgt, sep="\n", header=None)
    data = pd.concat([text_src, text_tgt], axis=1)
    data.columns = ["source", "target"]
    data = data.dropna(subset=["source", "target"])
    return data


def clean_and_filter(data: pd.DataFrame, max_len: int) -> pd.DataFrame:
    """Preprocess sentences and filter by max length."""
    data = data.copy()
    data["source"] = data["source"].apply(pp.preprocess_sentence)
    data["target"] = data["target"].apply(pp.preprocess_sentence)
    mask = (data["source"].str.split().str.len() <= max_len) & (
        data["target"].str.split().str.len() <= max_len
    )
    return data[mask].reset_index(drop=True)


def prepare_data():
    """Download, preprocess, and return train/test splits with vocabularies."""
    print("Loading training data...")
    train_data = load_parallel_data(
        "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en",
        "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi",
    )
    train_data = clean_and_filter(train_data, MAX_LEN)

    print("Loading test data...")
    test_data = load_parallel_data(
        "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en",
        "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi",
    )
    test_data = clean_and_filter(test_data, MAX_LEN)

    # Tokenize — build vocab only from training data (no data leakage)
    train_src = pp.preprocess_corpus(train_data["source"], TOKENIZERS[SOURCE_LANG], MIN_WORD_COUNT)
    train_tgt = pp.preprocess_corpus(train_data["target"], TOKENIZERS[TARGET_LANG], MIN_WORD_COUNT)

    src_vocab = pp.read_vocab(train_src)
    tgt_vocab = pp.read_vocab(train_tgt)

    # Tokenize test data (OOV words will map to UNK via indexes_from_sentence)
    test_src = pp.preprocess_corpus(test_data["source"], TOKENIZERS[SOURCE_LANG], MIN_WORD_COUNT)
    test_tgt = pp.preprocess_corpus(test_data["target"], TOKENIZERS[TARGET_LANG], MIN_WORD_COUNT)

    max_seq_length = MAX_LEN + 2  # +2 for SOS and EOS

    # Build tensors
    def build_tensors(src_sents, tgt_sents):
        pairs = [
            pp.tensors_from_pair(src_vocab, tgt_vocab, s, t, max_seq_length)
            for s, t in zip(src_sents, tgt_sents)
        ]
        x, y = zip(*pairs)
        x = torch.transpose(torch.cat(x, dim=-1), 1, 0)
        y = torch.transpose(torch.cat(y, dim=-1), 1, 0)
        return x, y

    x_train, y_train = build_tensors(train_src, train_tgt)
    x_test, y_test = build_tensors(test_src, test_tgt)

    print(
        f"Data loaded — Train: {len(train_src)}, Test: {len(test_src)}\n"
        f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}\n"
    )

    return (
        x_train, y_train, x_test, y_test,
        src_vocab, tgt_vocab,
        train_src, train_tgt, test_src, test_tgt,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    x_train, y_train, x_test, y_test,
    src_vocab, tgt_vocab,
    device,
):
    set_seed()

    model = nmt.Seq2Seq(len(src_vocab), len(tgt_vocab), HIDDEN_SIZE, N_LAYERS, device)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )
    cross_entropy = nn.CrossEntropyLoss()

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    total_batches = (len(x_train) + BATCH_SIZE - 1) // BATCH_SIZE
    indices = list(range(len(x_train)))
    best_loss = float("inf")

    print("Starting training...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in tqdm(
            enumerate(pp.batch_generator(indices, BATCH_SIZE)),
            desc=f"Epoch {epoch}/{EPOCHS}",
            total=total_batches,
        ):
            x = x_train[batch, :]
            y_teachf = y_train[batch, :-1]
            y_true = y_train[batch, 1:]

            optimizer.zero_grad()
            H = model.forward_train(x, y_teachf)
            loss = cross_entropy(H, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / total_batches
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} — loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nTraining finished. Best loss: {best_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
def compute_bleu(n: int, references, hypotheses) -> float:
    cc = SmoothingFunction()
    weights = [1.0 / n] * n + [0.0] * (4 - n)
    return corpus_bleu(references, hypotheses, weights, smoothing_function=cc.method1)


def test(x_test, y_test, src_vocab, tgt_vocab, test_src, test_tgt, device):
    model = nmt.Seq2Seq(len(src_vocab), len(tgt_vocab), HIDDEN_SIZE, N_LAYERS, device)
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()

    x_test = x_test.to(device)

    print("Running evaluation...")
    hypotheses = []
    with torch.no_grad():
        for i, x in tqdm(enumerate(x_test), desc="Testing", total=len(test_tgt)):
            y = model(x.unsqueeze(0))
            hypothesis = tgt_vocab.unindex_words(y[1:-1])
            hypotheses.append(hypothesis)

    bleu1 = compute_bleu(1, test_tgt, hypotheses)
    bleu4 = compute_bleu(4, test_tgt, hypotheses)
    print(f"\nBLEU-1: {bleu1:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

    # Show sample translations
    print("\nSample translations:")
    for i in range(11, min(21, len(test_src))):
        with torch.no_grad():
            y = model(x_test[i].unsqueeze(0))
        translation = " ".join(tgt_vocab.unindex_words(y[1:-1]))
        source = " ".join(test_src[i])
        print(f'  Source:      "{source}"')
        print(f'  Translation: "{translation}"\n')


# ---------------------------------------------------------------------------
# Interactive translation
# ---------------------------------------------------------------------------
def translate(src_vocab, tgt_vocab, device):
    model = nmt.Seq2Seq(len(src_vocab), len(tgt_vocab), HIDDEN_SIZE, N_LAYERS, device)
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()

    max_seq_length = MAX_LEN + 2

    print("Interactive translation (Ctrl+C to quit):\n")
    try:
        while True:
            sentence = input("Source: ")
            sentence = pp.preprocess_sentence(sentence)
            tokens = [w.lower() for w in TOKENIZERS[SOURCE_LANG](sentence)]
            tensor = pp.tensor_from_sentence(src_vocab, tokens, max_seq_length)
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                y = model(tensor)
            translation = " ".join(tgt_vocab.unindex_words(y[1:-1]))
            print(f'Translation: "{translation}"\n')
    except KeyboardInterrupt:
        print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Neural Machine Translation (EN → VI)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the model")
    subparsers.add_parser("test", help="Evaluate the model on test data")
    subparsers.add_parser("translate", help="Interactively translate sentences")

    args = parser.parse_args()
    device = get_device()

    (
        x_train, y_train, x_test, y_test,
        src_vocab, tgt_vocab,
        train_src, train_tgt, test_src, test_tgt,
    ) = prepare_data()

    if args.command == "train":
        train(x_train, y_train, x_test, y_test, src_vocab, tgt_vocab, device)
    elif args.command == "test":
        test(x_test, y_test, src_vocab, tgt_vocab, test_src, test_tgt, device)
    elif args.command == "translate":
        translate(src_vocab, tgt_vocab, device)


if __name__ == "__main__":
    main()
