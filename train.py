import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam, lr_scheduler
from math import inf, exp
import numpy as np
import time
from dataset import produce_datasets
from models import LanguageModelingRNN
from tensorboardX import SummaryWriter


num_epochs = 1000
init_lr = 1e-3
gamma = 0.5
decay_every = 10
bs = 5
l2_reg = 0
log_every = 1
val_ratio = 0.1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_file_path = ''  # insert conversation dataset file here


class CollatePad(object):

    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        """
        Collates groups of Tensors of different lengths into one Tensor (for each group).
        :param batch: list of tuple of tensors (e.g. list of 32 tuples, where the first elements are inputs and the
        seconds are targets). Individual Tensors must be of shape (T, N1, N2, N3, ...) where T is the variable dimension
        and N1, N2, N3, ... are any number of additional dimensions of fixed size. Inside the final Tensor, elements
        will be sorted in descending order of length. IMPORTANT: it is assumed that the order according to length is
        consistent across groups, i.e. the variable lengths inside the different groups of Tensor are equal along the
        same index in the group. Example: group 1 represents tokenized sentences [["Hi", "Paolo", speaking"],
        ["Hi", "how", "are", "you", "today"]], group 2 POS tags [[EXCL, NOM, VRB], [EXCL, ADV, BE, SUBJ, ADV]],
        lengths are consistent across groups thus the program can sort according to the length of item of any attribute.
        :return: tuple of:
            * list of Tensors containing the padded batch, one for each group;
            * list of Tensors containing the variable lengths, one for each group.
        """

        batch = sorted(batch, reverse=True, key=lambda elem: len(elem[0]))
        variable_lengths = np.array([[tensor.shape[0] for tensor in tensors] for tensors in batch])
        batch_size, num_tensors = variable_lengths.shape
        max_lengths = np.max(variable_lengths, axis=0)

        padded_batch = []
        lengths = []
        for group_index, max_length in enumerate(max_lengths):
            tensors = [tensors[group_index] for tensors in batch]
            single_tensor_shape = (max_length, *batch[0][group_index].shape[1:])
            padded_tensor = self.pad_value * torch.ones((batch_size, *single_tensor_shape), dtype=torch.long)
            for batch_index, length in enumerate(variable_lengths[:, group_index]):
                padded_tensor[batch_index, :length] = tensors[batch_index]
            padded_batch.append(padded_tensor)
            lengths.append(torch.tensor(variable_lengths[:, group_index], dtype=torch.int))
        return padded_batch, lengths


if __name__ == '__main__':

    print("Loading data...")
    ds_train, ds_val = produce_datasets(dataset_file_path, max_lexicon_size=lexicon_size)
    padding_idx = ds_train.label_map.label_to_index['<PAD>']
    collate_fn = CollatePad(padding_idx)
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=bs, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=bs, collate_fn=collate_fn)
    print("done. (train set size = {}, lexicon size = {})".format(len(ds_train), len(ds_train.label_map)))

    print("Loading model...", end='')
    model = LanguageModelingRNN(lexicon_size=len(ds_train.label_map), embedding_dim=64, padding_idx=padding_idx,
                                lstm_layers=2, hidden_size=512, p_dropout=0.5, dev=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=init_lr, weight_decay=l2_reg)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_every, gamma=gamma)
    criterion = NLLLoss(reduction='mean', ignore_index=padding_idx)
    print('done. (parameter count: {})'.format(sum(p.numel() for p in model.parameters())))

    print("Starting training...")
    writer = SummaryWriter()
    best_val_loss = inf
    for epoch in range(num_epochs):

        model.train()
        t_start = time.time()
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(train_loader):

            optimizer.zero_grad()

            # move to GPU
            tokens = tokens.to(device)
            targets = targets.to(device)

            # forward
            output = model(tokens, input_lengths)
            loss = criterion(output, targets)
            loss_value = loss.item()

            # backward
            loss.backward()

            # update step
            optimizer.step()

            if (j + 1) % log_every == 0:
                writer.add_scalar("Loss/train", loss_value, global_step=epoch*len(train_loader) + j)
                print("\rEpoch %3d/%3d, loss: %2.6f, "
                      "batch: %3d/%3d, pad length: %4d, lr: %.6f" % (epoch + 1, num_epochs, loss_value, j + 1,
                                                                     len(train_loader), max(input_lengths),
                                                                     scheduler.get_lr()[0]), end='')

        # evaluation
        epoch_duration = time.time() - t_start
        print("\nEpoch completed in {:3.2f}s. Evaluating...\r".format(epoch_duration), end='')
        model.eval()
        val_loss = 0
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(val_loader):

            # move to GPU
            tokens = tokens.to(device)
            targets = targets.to(device)

            # forward
            output = model(tokens, input_lengths)
            loss = criterion(output, targets)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, global_step=(epoch + 1) * len(train_loader))
        writer.add_embedding(mat=model.embedding.weight.data, metadata=ds_train.label_map.label_to_index.keys(),
                             global_step=(epoch + 1) * len(train_loader))
        print("Evaluation completed. Validation loss: {:2.6f}, average perplexity: {:2.6f}".format(val_loss,
                                                                                                   exp(val_loss)))
        # save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, "val_{:.4f}.pt".format(val_loss))

        scheduler.step(epoch)
