from loader import produce_conversations
import torch

meta_tokens = {"<PAD>", "<BEGIN>", "<CHANGE SENDER>", "<NEW MESSAGE>", "<END>"}


class LabelIndexMap(object):
    """
    Class used to remap N labels from cluttered, arbitrary values to non-negative, contiguous values in range [0, N-1].
    """

    def __init__(self, all_labels, sort_key=None, required_mappings=None):
        assert (sort_key is None or required_mappings is None), "use only one among sort_key, predefined_positions"
        self.individual_labels = list(set(all_labels))
        if sort_key is not None:
            self.individual_labels = sorted(all_labels, key=sort_key)
        if required_mappings is not None:
            # set items to the required positions
            for element, required_position in required_mappings.items():
                if required_position >= len(self.individual_labels):
                    raise ValueError("Required position is out of range")
                if self.individual_labels[required_position] in required_mappings:
                    raise ValueError("Incompatible required mapping: "
                                     "label {} and {} both required "
                                     "in position {}".format(element, self.individual_labels[required_position],
                                                             required_position))

                # swap
                current_position = self.individual_labels.index(element)
                self.individual_labels[current_position] = self.individual_labels[required_position]
                self.individual_labels[required_position] = element

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

        # transform to Tensor
        conversation_tokens = torch.tensor([self.label_map[token] for token in conversation_tokens])
        target_tokens = torch.tensor([self.label_map[token] for token in target_tokens])
        return conversation_tokens, target_tokens

    def __len__(self):
        return len(self._convs)


def produce_datasets(whatsapp_export_filepath, val_ratio=0.2):
    conversations = produce_conversations(whatsapp_export_filepath)
    tokens = set(token for conv in conversations for message in conv for token in message['Tokens']) | meta_tokens
    label_map = LabelIndexMap(tokens, required_mappings={"<PAD>": 0})
    label_map.save("legend.txt")

    train_ds_size = int((1 - val_ratio) * len(conversations))
    train_conversations, val_conversations = conversations[:train_ds_size], conversations[train_ds_size:]
    return WhatsappConversationDataset(train_conversations, label_map), \
           WhatsappConversationDataset(val_conversations, label_map)
