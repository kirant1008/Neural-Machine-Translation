    
import torch.nn.init as init

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

class enc_rnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=100):
        super(enc_rnn, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(
            hidden_size,
            int(hidden_size/2),  # Bi-directional processing will ouput vectors of double size, therefore I reduced output dimensionality
            num_layers=n_layers,
            batch_first=True,  # First dimension of input tensor will be treated as a batch dimension
            bidirectional=True
        )

    # word_inputs: (batch_size, seq_length), h: (h_or_c, layer_n_direction, batch, seq_length)
    def forward(self, word_inputs, hidden):         
        # embedded (batch_size, seq_length, hidden_size)
        embedded = self.embedding(word_inputs)
        # output (batch_size, seq_length, hidden_size*directions)
        # hidden (h: (num_layers*directions, batch_size, hidden_size),
        #         c: (num_layers*directions, batch_size, hidden_size))
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batches):
      h_s = torch.zeros(self.n_layers*2, batches, int(self.hidden_size/2)).cuda()
      c_s = torch.zeros(self.n_layers*2, batches, int(self.hidden_size/2)).cuda()
      hidden = (h_s, c_s)
      return hidden

class dec_rnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=100):
        super(dec_rnn, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)

    def forward(self, word_inputs, hidden):
        # Note: we run this one by one
        # embedded (batch_size, 1, hidden_size)
        embedded = self.embedding(word_inputs).unsqueeze_(1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

class seq2seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, n_layers):
        super(seq2seq, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.encoder = enc_rnn(input_vocab_size, hidden_size, self.n_layers)
        self.decoder = dec_rnn(output_vocab_size, hidden_size, self.n_layers)

        self.lin = nn.Linear(hidden_size, output_vocab_size)
        init.normal_(self.lin.weight, 0.0, 0.2)

        self.softmax = nn.Softmax()

    def _forward_encoder(self, x):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(x, init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        self.decoder_hidden_h = encoder_hidden_h.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        self.decoder_hidden_c = encoder_hidden_c.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        return self.decoder_hidden_h.cuda(), self.decoder_hidden_c.cuda()

    def forward_train(self, x, y):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        H = []
        for i in range(y.shape[1]):
            input = y[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.lin(decoder_output.squeeze(1))
            # h: (batch_size, vocab_size, 1)
            H.append(h.unsqueeze(2))

        # H: (batch_size, vocab_size, seq_len)
        return torch.cat(H, dim=2)

    def forward(self, x):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        current_y = SOS_idx
        result = [current_y]
        counter = 0
        while current_y != EOS_idx and counter < 100:
            input = torch.tensor([current_y]).cuda()
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (vocab_size)
            h = self.lin(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=0)
            current_y = current_y.item()
            result.append(current_y)
            counter += 1

        return result

#reference : https://tsdaemon.github.io/2018/07/08/nmt-with-pytorch-encoder-decoder.html