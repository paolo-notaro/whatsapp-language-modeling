# whatsapp-language-modeling
A language model for Whatsapp conversations, with LSTM-based architectures in Pytorch.

## General

This project aims at modeling instant messaging chats between two users of Whatsapp using Deep Learning. In particular, the focus 
is on being able to produce realistic conversations rather than just realistic messages, i.e. modeling also how the users
alternate their message in the conversation time window. To this end, A single conversation represents a corpus, a consecutive exchange of messages represent a training instance (or document). Words and punctuation are the fundamental tokens. Change of user, end of message and end of conversation are all modeled through corresponding metatokens.
Moreover, emoji are also introduced in the project-level dictionary of the algorithm as valid as any other word or punctuation symbol. These new aspects require some additional effort from the neural network to effectively understand how a realistic exchange of message of Whatsapp occurs.

## Data

Input data of the system are Whatsapp conversation exports, as they are downloadable from the mobile version of the app in Android.
From this export the loader system is able to segment conversation, create the token dictionary and train a custom
langage predition/generation model.
