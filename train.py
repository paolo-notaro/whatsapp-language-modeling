import torch
from torch.utils.data import DataLoader
from torch.nn import Embedding, LSTM, Linear, Module, NLLLoss, functional as func
from torch.optim import Adam
from conversation_dataset import WhatsappConversationDataset

num_epochs = 1000
lr = 1e-3
bs = 32
log_every = 1
save_every = 1
criterion = NLLLoss(reduction='mean')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_file_path = '/home/paulstpr/Downloads/WhatsApp Chat with Sara Pontelli 💙.txt'


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


if __name__ == '__main__':

    print("Loading dataset...")
    ds = WhatsappConversationDataset(dataset_file_path)
    train_loader = DataLoader(ds, shuffle=True, batch_size=bs, collate_fn=lambda x: x)

    print("Loading model...")
    model = LanguageModelingRNN(lexicon_size=ds.num_tokens, embedding_dim=128,
                                lstm_layers=2, hidden_size=512, fc_hidden_size=2560, dev=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        for j, batch in enumerate(train_loader):

            optimizer.zero_grad()
            batch_loss = 0
            for tokens, targets in batch:

                # move to GPU
                tokens = tokens.to(device)
                targets = targets.to(device)

                # forward
                output = model(tokens).permute(0, 2, 1)
                loss = criterion(output, targets)
                batch_loss += loss.item()

                # backward
                loss.backward()

            batch_loss /= len(batch)

            # update step
            optimizer.step()

            if (j + 1) % log_every == 0:
                print("\rEpoch %3d, loss: %2.6f, batch: %3d/%3d" % (epoch + 1, batch_loss, j + 1, len(train_loader)),
                      end='')

        if epoch % save_every == 0:
            torch.save(model, "model.pt")

        print("")
