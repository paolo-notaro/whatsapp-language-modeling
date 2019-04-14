import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam
from conversation_dataset import WhatsappConversationDataset
from models import LanguageModelingRNN

num_epochs = 1000
lr = 1e-3
bs = 32
log_every = 1
save_every = 1
criterion = NLLLoss(reduction='mean')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_file_path = '/home/paulstpr/Downloads/WhatsApp Chat with Sara Pontelli 💙.txt'


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
