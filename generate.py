import torch
import numpy as np
from train import LanguageModelingRNN
from string import punctuation

yes = ['y', 'yes']
no = ['n', 'no']

model_name = "val_4.5134.pt"


def post_process(conv):
    res = "0:"
    speaker = 0
    capitalize_next = False
    for token in conv:
        if token == "<NEW MESSAGE>":
            res += "\n{}:".format(speaker)
            capitalize_next = True
        elif token == "<CHANGE SENDER>":
            speaker = int(not speaker)
            last_sender_switch = res.rfind(':') - 1
            res = res[:last_sender_switch] + str(speaker) + res[last_sender_switch + 1:]
        elif token not in ("<BEGIN>", "<END>"):
            if token not in punctuation:
                res += " "
            if capitalize_next:
                token = token[0].upper() + token[1:]
                capitalize_next = False
            res += token

    if res[0] == " ":
        res = res[1:]
    if res.endswith(":"):
        res = res[:-3]
    return res


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(model_name)
    model.to(device)
    model.eval()

    legend_file_content = np.loadtxt("legend.txt", dtype=str, delimiter="\t", comments=None)
    legend = {entry[0]: int(entry[1]) for entry in legend_file_content}
    legend_reversed = {int(entry[1]): entry[0] for entry in legend_file_content}

    done = False
    while not done:

        valid = False
        resp = None
        while not valid:
            resp = input('Generate conversation?')
            if resp.lower() in (yes + no):
                valid = True

        if resp in yes:

            print("Generating...")
            conversation = []
            model.reset_state()
            curr_token = "<BEGIN>"
            while curr_token != '<END>':

                curr_token_tensor = torch.tensor(legend[curr_token]).to(device).unsqueeze(0).unsqueeze(0)
                output_probabilities = model(curr_token_tensor, [1], reset_state=False).squeeze().cpu().detach().numpy()

                next_token_index = np.random.choice(len(legend), p=np.exp(output_probabilities))
                next_token = legend_reversed[next_token_index]

                conversation.append(next_token)
                curr_token = next_token

            conversation = post_process(conversation)
            print(conversation)

        else:
            done = True
