import re
import datetime
from emoji import UNICODE_EMOJI
from numpy import random
from string import punctuation
from random import shuffle


def produce_conversations(file_path, word_emoji_tokenization=True):

    with open(file_path, 'r', encoding='utf8') as f:
        rows = re.findall(r'(\d+.\d+.\d+,\s+\d+:\d{2})\s-\s(.*):\s(.*)', f.read())
        msg_table = [{key: value for key, value in zip(('DateTime', 'Sender', 'Message'), row)} for row in rows]

    conversations = []
    conversation = []
    skip_count = 0
    last_timestamp = datetime.datetime.min
    for index, row in enumerate(msg_table):

        print("Processing message %d/%d\r" % (index + 1, len(msg_table)), end='')

        # remove multiple spaces and tabs
        row['Message'] = re.sub(' +', ' ', row['Message'])
        row['Message'] = row['Message'].replace("\t", "")

        # tokenization
        if row['Message'] == "<Media omitted>":
            row['Tokens'] = ["<Media omitted>"]
        else:
            # word/emoji tokenization
            if word_emoji_tokenization:
                tokens = []
                token_start_index = 0
                for i, char in enumerate(row['Message'] + " "):
                    if char in UNICODE_EMOJI or char in punctuation or char == " ":
                        token = row['Message'][token_start_index:i]
                        if token != "":
                            tokens.append(token.lower())
                        if char in UNICODE_EMOJI or char in punctuation:
                            tokens.append(char)
                        token_start_index = i+1
                row['Tokens'] = tokens
            else:
                row['Tokens'] = list(row['Message'])

        # divide into sequences
        timestamp = datetime.datetime.strptime(row['DateTime'], "%m/%d/%y, %H:%M")
        if timestamp > last_timestamp + datetime.timedelta(minutes=90):
            if len(conversation) > 3:
                conversations.append(conversation)
            else:
                skip_count += 1
            conversation = []

        conversation.append(row)
        last_timestamp = timestamp
    print("")
    shuffle(conversations)
    return conversations


if __name__ == '__main__':

    convs = produce_conversations('', True)
    print([len(conversation) for conversation in convs])
    print(convs[random.choice(len(convs))])
