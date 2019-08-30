# whatsapp-language-modeling
Language Modeling of Whatsapp conversations with LSTM-based architectures in Pytorch.

## General

This project aims at modeling instant messaging chats between two users of Whatsapp using Deep Learning. In particular, the focus 
is on being able to produce realistic conversations rather than just realistic messages, i.e. modeling also how the users
alternate their message in the conversation time window. Moreover, emoji are introduced in the project-level dictionary of the
algorithm as actual tokens, as valid as any other word or punctucation symbols. These two aspect mean some additional effort is 
required from the neural network to effectively understand how a realistic exchange of message of Whatsapp occurs.

## Data

Input data of the system are Whatsapp conversation exports, as they are downloadable from the mobile version of the app.
From this export the loader system is able to segment conversation, create the token dictionary and train a custom
langage predition/generation model.
