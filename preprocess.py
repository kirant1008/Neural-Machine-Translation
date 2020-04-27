import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import time

SOS_token = '<start>'
EOS_token = '<end>'
UNK_token = '<unk>'
PAD_token = '<pad>'

SOS_idx = 0
EOS_idx = 1
UNK_idx = 2
PAD_idx = 3

#class vocab is used to generate word to index and index to word
class Vocab:
    def __init__(self):
        self.index2word = {
            SOS_idx: SOS_token,
            EOS_idx: EOS_token,
            UNK_idx: UNK_token,
            PAD_idx: PAD_token
        }
        self.word2index = {v: k for k, v in self.index2word.items()}

    def index_words(self, words):
        for word in words:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            n_words = len(self)
            self.word2index[word] = n_words
            self.index2word[n_words] = word

    def __len__(self):
        assert len(self.index2word) == len(self.word2index)
        return len(self.index2word)

    def unidex_words(self, indices):
        return [self.index2word[i] for i in indices]

    def to_file(self, filename):
        values = [w for w, k in sorted(list(self.word2index.items())[5:])]
        with open(filename, 'w') as f:
            f.write('\n'.join(values))

    @classmethod
    def from_file(cls, filename):
        vocab = Vocab()
        with open(filename, 'r') as f:
            words = [l.strip() for l in f.readlines()]
            vocab.index_words(words)


#it is used to help in tokenization of sentence
def preprocess_func(sents, tokenizer, min_word_count):
    n_words = {}

    sents_tokenized = []
    for sent in sents:
        sent_tokenized = [w.lower() for w in tokenizer(sent)]

        sents_tokenized.append(sent_tokenized)

        for word in sent_tokenized:
            if word in n_words:
                n_words[word] += 1
            else:
                n_words[word] = 1

    for i, sent_tokenized in enumerate(sents_tokenized):
        sent_tokenized = [t if n_words[t] >= min_word_count else UNK_token for t in sent_tokenized]
        sents_tokenized[i] = sent_tokenized

    return sents_tokenized

#it is used to read the vocab from the sentences
def read_vocab(sents):
    vocab = Vocab()
    for sent in sents:
        vocab.index_words(sent)

    return vocab

#it used to generate words from sentences
def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

#used to append eos_idx and sos_idx to the sentence and pad the sentence to max_seq_LEN 
def tsr_sent(vocab, sentence, max_seq_length):
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_idx)
    indexes.insert(0, SOS_idx)
    # we need to have all sequences the same length to process them in batches
    if len(indexes) < max_seq_length:
        indexes += [PAD_idx] * (max_seq_length - len(indexes))
    tensor = torch.LongTensor(indexes)
    return tensor

def tensrs_prs(source_vocab,target_vocab,source_sent, target_sent, max_seq_length):
    source_tensor = tsr_sent(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    target_tensor = tsr_sent(target_vocab, target_sent, max_seq_length).unsqueeze(1)
    return (source_tensor, target_tensor)

def unicode_to_ascii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sent):
    sent = unicode_to_ascii(sent.lower().strip())
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)
    sent = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sent)
    
    return sent

import math

#used to generate batches 
def batch_generator(batch_indices, batch_size):
    batches = math.ceil(len(batch_indices)/batch_size)
    for i in range(batches):
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        if batch_end > len(batch_indices):
            yield batch_indices[batch_start:]
        else:
            yield batch_indices[batch_start:batch_end]

#this function is same as above but it is used in translate function in nmt.py
def tensors_from_pair_translate(source_vocab,source_sent,max_seq_length):
    source_tensor = tensor_from_sentence_translate(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    return source_tensor

#used in translate function
def tensor_from_sentence_translate(vocab, sentence, max_seq_length):
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_idx)
    indexes.insert(0, SOS_idx)
    # we need to have all sequences the same length to process them in batches
    if len(indexes) < max_seq_length:
        indexes += [PAD_idx] * (max_seq_length - len(indexes))
    tensor = torch.LongTensor(indexes)
    return tensor

#used in translate function
def preprocess_corpus_translate(sents, tokenizer):

    sents_tokenized = []
    sent_tokenized = [w.lower() for w in tokenizer(sents)]

    sents_tokenized.append(sent_tokenized)

    return sents_tokenized

#reference : https://tsdaemon.github.io/2018/07/08/nmt-with-pytorch-encoder-decoder.html