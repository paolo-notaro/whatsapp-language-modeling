import torch
from torch.nn import Embedding, LSTM, Linear, Module, functional as func


class LanguageModelingRNN(Module):

    def __init__(self, lexicon_size, embedding_dim, lstm_layers, hidden_size, fc_hidden_size, dev):
        super().__init__()

        self.hidden_state = None
        self.hidden_size = hidden_size
        self.device = dev
        self.lstm_layers = lstm_layers
        self.reset_state()

        self.embedding = Embedding(lexicon_size, embedding_dim)
        self.lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc1 = Linear(hidden_size, fc_hidden_size)
        self.fc2 = Linear(fc_hidden_size, lexicon_size)

    def forward(self, x, reset_state=True):

        if reset_state:
            self.reset_state()

        x = self.embedding(x)
        z, self.hidden_state = self.lstm(x, self.hidden_state)
        x = func.relu(self.fc1(z))
        return func.log_softmax(self.fc2(x), dim=2)

    def reset_state(self):
        self.hidden_state = [torch.zeros((self.lstm_layers, 1, self.hidden_size)).to(self.device) for _ in range(2)]