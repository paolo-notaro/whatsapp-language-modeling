from loader import produce_conversations
import torch

meta_tokens = {"<BEGIN>", "<CHANGE SENDER>", "<NEW MESSAGE>", "<END>"}


class LabelIndexMap(object):
    """
    class used to remap N labels from cluttered, arbitrary values to non-negative, contiguous values in range [0, N-1].
    """

    def __init__(self, all_labels, sort_key=None):
        self.individual_labels = sorted(set(all_labels), key=sort_key)
        self.label_to_index = {label: self.individual_labels.index(label) for label in self.individual_labels}
        self.index_to_label = {self.individual_labels.index(label): label for label in self.individual_labels}

    def __getitem__(self, item):
        return self.label_to_index[item]

    def __len__(self):
        return len(self.label_to_index)

    def save(self, filename):
        with open(filename, "w") as f:
            for item in self.label_to_index.items():
                f.write("{}\t{}\n".format(*item))


class WhatsappConversationDataset:

    def __init__(self, conversations: list, label_map: LabelIndexMap):

        super().__init__()
        self._convs = conversations
        self.label_map = label_map

    def __getitem__(self, idx):
        conversation = self._convs[idx]
        conversation_tokens = ["<BEGIN>"]
        previous_message_sender = conversation[0]['Sender']
        for message in conversation:
            message_tokens = []
            if message['Sender'] != previous_message_sender:
                message_tokens += ["<CHANGE SENDER>"]
            message_tokens += [token for token in message['Tokens']]
            message_tokens += ["<NEW MESSAGE>"]
            conversation_tokens.extend(message_tokens)
            previous_message_sender = message['Sender']
        target_tokens = conversation_tokens[1:] + ["<END>"]

        conversation_tokens = torch.tensor([self.label_map[token] for token in conversation_tokens])
        target_tokens = torch.tensor([self.label_map[token] for token in target_tokens])
        return conversation_tokens.unsqueeze(0), target_tokens.unsqueeze(0)

    def __len__(self):
        return len(self._convs)


def produce_datasets(whatsapp_export_filepath, val_ratio=0.2):
    conversations = produce_conversations(whatsapp_export_filepath)
    tokens = set(token for conv in conversations for message in conv for token in message['Tokens']).union(meta_tokens)
    label_map = LabelIndexMap(tokens)
    label_map.save("legend.txt")

    train_ds_size = int((1 - val_ratio) * len(conversations))
    train_conversations, val_conversations = conversations[:train_ds_size], conversations[train_ds_size:]
    return WhatsappConversationDataset(train_conversations, label_map), \
           WhatsappConversationDataset(val_conversations, label_map)
