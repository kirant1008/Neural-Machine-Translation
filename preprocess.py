import math
import re
import unicodedata
from typing import Callable, List, Optional

import torch

SOS_token = "<start>"
EOS_token = "<end>"
UNK_token = "<unk>"
PAD_token = "<pad>"

SOS_idx = 0
EOS_idx = 1
UNK_idx = 2
PAD_idx = 3


class Vocab:
    """Maps words to integer indices and back."""

    def __init__(self):
        self.index2word = {
            SOS_idx: SOS_token,
            EOS_idx: EOS_token,
            UNK_idx: UNK_token,
            PAD_idx: PAD_token,
        }
        self.word2index = {v: k for k, v in self.index2word.items()}

    def index_words(self, words: List[str]):
        for word in words:
            self.index_word(word)

    def index_word(self, word: str):
        if word not in self.word2index:
            n_words = len(self)
            self.word2index[word] = n_words
            self.index2word[n_words] = word

    def __len__(self):
        assert len(self.index2word) == len(self.word2index)
        return len(self.index2word)

    def unindex_words(self, indices: List[int]) -> List[str]:
        return [self.index2word[i] for i in indices]

    def to_file(self, filename: str):
        # Skip the 4 special tokens (SOS, EOS, UNK, PAD)
        values = [w for w, k in sorted(list(self.word2index.items())[4:])]
        with open(filename, "w") as f:
            f.write("\n".join(values))

    @classmethod
    def from_file(cls, filename: str) -> "Vocab":
        vocab = Vocab()
        with open(filename, "r") as f:
            words = [line.strip() for line in f.readlines()]
            vocab.index_words(words)
        return vocab


def preprocess_corpus(
    sents: List[str],
    tokenizer: Callable,
    min_word_count: int,
) -> List[List[str]]:
    """Tokenize sentences and replace rare words with UNK."""
    n_words: dict[str, int] = {}

    sents_tokenized = []
    for sent in sents:
        sent_tokenized = [w.lower() for w in tokenizer(sent)]
        sents_tokenized.append(sent_tokenized)
        for word in sent_tokenized:
            n_words[word] = n_words.get(word, 0) + 1

    for i, sent_tokenized in enumerate(sents_tokenized):
        sents_tokenized[i] = [
            t if n_words[t] >= min_word_count else UNK_token
            for t in sent_tokenized
        ]

    return sents_tokenized


def read_vocab(sents: List[List[str]]) -> Vocab:
    """Build a Vocab from tokenized sentences."""
    vocab = Vocab()
    for sent in sents:
        vocab.index_words(sent)
    return vocab


def indexes_from_sentence(vocab: Vocab, sentence: List[str]) -> List[int]:
    """Convert words to indices, mapping OOV words to UNK."""
    return [vocab.word2index.get(word, UNK_idx) for word in sentence]


def tensor_from_sentence(
    vocab: Vocab, sentence: List[str], max_seq_length: int
) -> torch.LongTensor:
    """Convert a sentence to a padded LongTensor with SOS/EOS markers."""
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.insert(0, SOS_idx)
    indexes.append(EOS_idx)
    if len(indexes) < max_seq_length:
        indexes += [PAD_idx] * (max_seq_length - len(indexes))
    return torch.LongTensor(indexes)


def tensors_from_pair(
    source_vocab: Vocab,
    target_vocab: Vocab,
    source_sent: List[str],
    target_sent: List[str],
    max_seq_length: int,
) -> tuple:
    """Convert a source/target sentence pair to padded tensors."""
    source_tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    target_tensor = tensor_from_sentence(target_vocab, target_sent, max_seq_length).unsqueeze(1)
    return (source_tensor, target_tensor)


def unicode_to_ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(sent: str) -> str:
    """Lowercase, normalize, and clean a sentence."""
    sent = unicode_to_ascii(sent.lower().strip())
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)
    sent = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sent)
    return sent


def batch_generator(batch_indices: list, batch_size: int):
    """Yield successive batches from a list of indices."""
    batches = math.ceil(len(batch_indices) / batch_size)
    for i in range(batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        if batch_end > len(batch_indices):
            yield batch_indices[batch_start:]
        else:
            yield batch_indices[batch_start:batch_end]
