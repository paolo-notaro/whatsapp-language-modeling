import torch
from torch.nn import Embedding, LSTM, Linear, Module, Sequential, Dropout, functional as func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def kaiming_init(m):
    if isinstance(m, Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, LSTM):
        for name, param in m._parameters.items():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)
            elif "bias" in name:
                torch.nn.init.constant_(param, 0)


class LanguageModelingRNN(Module):

    def __init__(self, lexicon_size, embedding_dim, padding_idx, lstm_layers, hidden_size, p_dropout, dev):
        super().__init__()

        self.hidden_state = None
        self.hidden_size = hidden_size
        self.device = dev
        self.lstm_layers = lstm_layers

        self.embedding = Embedding(num_embeddings=lexicon_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = Sequential(Dropout(p=p_dropout, inplace=True), Linear(hidden_size, lexicon_size))

        self.reset_state()
        self.apply(kaiming_init)

    def forward(self, x, x_lengths, reset_state=True):

        if reset_state:
            self.reset_state(bs=len(x_lengths))

        x = self.embedding(x)

        x = pack_padded_sequence(x, x_lengths, batch_first=True)
        z, self.hidden_state = self.lstm(x, self.hidden_state)
        z, _ = pad_packed_sequence(z, batch_first=True)

        z = self.fc(z)
        logits = func.log_softmax(z, dim=2)
        return logits.permute(0, 2, 1)

    def reset_state(self, bs=1):
        self.hidden_state = [torch.zeros((self.lstm_layers, bs, self.hidden_size)).to(self.device) for _ in range(2)]
