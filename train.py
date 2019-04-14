import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam
from math import inf, exp
from dataset import produce_datasets
from models import LanguageModelingRNN

num_epochs = 100
lr = 1e-3
bs = 32
log_every = 1
val_ratio = 0.1
criterion = NLLLoss(reduction='mean')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_file_path = '/home/paulstpr/Downloads/WhatsApp Chat with Sara Pontelli 💙.txt'


if __name__ == '__main__':

    ds_train, ds_val = produce_datasets(dataset_file_path)
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=bs, collate_fn=lambda x: x)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=1, collate_fn=lambda x: x[0])

    print("Loading model...", end='')
    model = LanguageModelingRNN(lexicon_size=len(ds_train.label_map), embedding_dim=128,
                                lstm_layers=2, hidden_size=512, fc_hidden_size=2560, dev=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    print('done.')

    print("Starting training...")
    best_val_loss = inf
    for epoch in range(num_epochs):

        model.train()
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
                print("\rEpoch %3d/%3d, loss: %2.6f, batch: %3d/%3d" % (epoch + 1, num_epochs, batch_loss, j + 1,
                                                                        len(train_loader)), end='')

        # evaluation
        print("\nEvaluating...\r", end='')
        model.eval()
        val_loss = 0
        for j, (tokens, targets) in enumerate(val_loader):

            # move to GPU
            tokens = tokens.to(device)
            targets = targets.to(device)

            # forward
            output = model(tokens).permute(0, 2, 1)
            loss = criterion(output, targets)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        print("Evaluation completed. Validation loss: {:2.6f}, average perplexity: {:2.6f}".format(val_loss,
                                                                                                   exp(val_loss)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, "val_{:.4f}.pt".format(val_loss))
