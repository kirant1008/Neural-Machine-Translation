import torch
import torch.nn as nn
import torch.nn.init as init

SOS_idx = 0
EOS_idx = 1


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,  # bidirectional doubles the output size
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, word_inputs, hidden):
        embedded = self.embedding(word_inputs)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size // 2, device=device)
        c = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size // 2, device=device)
        return (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=n_layers, batch_first=True
        )

    def forward(self, word_inputs, hidden):
        embedded = self.embedding(word_inputs).unsqueeze_(1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        hidden_size: int,
        n_layers: int,
        device: torch.device,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device

        self.encoder = Encoder(input_vocab_size, hidden_size, n_layers)
        self.decoder = Decoder(output_vocab_size, hidden_size, n_layers)

        self.fc_out = nn.Linear(hidden_size, output_vocab_size)
        init.normal_(self.fc_out.weight, 0.0, 0.2)

        self.softmax = nn.Softmax(dim=-1)

    def _forward_encoder(self, x):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size, self.device)
        _, encoder_hidden = self.encoder(x, init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        decoder_hidden_h = (
            encoder_hidden_h.permute(1, 0, 2)
            .reshape(batch_size, self.n_layers, self.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        decoder_hidden_c = (
            encoder_hidden_c.permute(1, 0, 2)
            .reshape(batch_size, self.n_layers, self.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        return decoder_hidden_h, decoder_hidden_c

    def forward_train(self, x, y):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        H = []
        for i in range(y.shape[1]):
            token = y[:, i]
            decoder_output, (decoder_hidden_h, decoder_hidden_c) = self.decoder(
                token, (decoder_hidden_h, decoder_hidden_c)
            )
            h = self.fc_out(decoder_output.squeeze(1))
            H.append(h.unsqueeze(2))

        # (batch_size, vocab_size, seq_len)
        return torch.cat(H, dim=2)

    def forward(self, x):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        current_y = SOS_idx
        result = [current_y]
        for _ in range(100):
            token = torch.tensor([current_y], device=self.device)
            decoder_output, (decoder_hidden_h, decoder_hidden_c) = self.decoder(
                token, (decoder_hidden_h, decoder_hidden_c)
            )
            h = self.fc_out(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=0)
            current_y = current_y.item()
            result.append(current_y)
            if current_y == EOS_idx:
                break

        return result
