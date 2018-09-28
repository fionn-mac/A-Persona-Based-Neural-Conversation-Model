from os import path
from io import open
import unicodedata
import re

import torch

import numpy as np
import pandas as pd

class Data_Preprocess(object):
    def __init__(self, path, min_length=5, max_length=20):
        self.path = path
        self.PAD_token = 0
        self.EOS_token = 2
        self.min_length = min_length
        self.max_length = max_length
        self.vocab = set("<EOS>")
        self.word2index = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>"]
        self.word2count = {}
        self.vocab_size = 3
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.train_lengths = []
        self.val_lengths = []
        self.dialogue_cols = ['dialogue1', 'dialogue2']
        self.use_cuda = torch.cuda.is_available()
        self.run()

    def load_vocabulary(self):
        with open(path.join(self.path, 'vocabulary.txt'), encoding='utf-8') as f:
            for word in f:
                word = word.strip('\n')
                self.vocab.add(word)
                self.index2word.append(word)
                self.word2index[word] = self.vocab_size
                self.vocab_size += 1

    def load_dialogues(self):
        # Load training and test set
        train_df = pd.read_csv(path.join(self.path, 'train.txt'), sep='|')
        val_df = pd.read_csv(path.join(self.path, 'val.txt'), sep='|')

        lengths = []

        # Iterate over dialogues of both training and test datasets
        for dataset in [train_df, val_df]:
            for index, row in dataset.iterrows():
                # Iterate through the text of both the dialogues of the row
                for dialogue in self.dialogue_cols:
                    d2n = []  # Dialogue Numbers Representation
                    for i, word_id in enumerate(row[dialogue].split()):
                        ''' No concept of Person ID during Pre-Training '''

                        effective_id = int(word_id) + 2
                        word = self.index2word[effective_id]

                        d2n.append(effective_id)
                        if word not in self.word2count:
                            self.word2count[word] = 0
                        self.word2count[word] += 1

                    # Replace |questions as word| to |question as number| representation
                    # Add <EOS> token at end of dialogue.
                    dataset.at[index, dialogue] = d2n + [self.EOS_token]

        return train_df, val_df

    def len_check(self, x, y):
        len_x = len(x)
        len_y = len(y)

        if len_x >= self.min_length and len_x <= self.max_length and \
           len_y >= self.min_length and len_y <= self.max_length:
           return True

        return False

    def filter_and_tensor(self, pairs):
        cleaned_pairs = [[], []]
        lengths = []

        for i, tup in enumerate(pairs):
            if not self.len_check(tup[0], tup[1]):
                continue

            cleaned_pairs[0].append(torch.LongTensor(tup[0]))
            cleaned_pairs[1].append(torch.LongTensor(tup[1]))

            lengths.append(len(cleaned_pairs[0][-1]))

        return cleaned_pairs[0], cleaned_pairs[1], lengths

    def sort_filter_tensor(self):
        xy_train = sorted(zip(self.x_train, self.y_train), key=lambda tup: len(tup[0]), reverse=True)
        xy_val = sorted(zip(self.x_val, self.y_val), key=lambda tup: len(tup[0]), reverse=True)

        self.x_train, self.y_train, self.train_lengths = self.filter_and_tensor(xy_train)
        self.x_val, self.y_val, self.val_lengths = self.filter_and_tensor(xy_val)

    def run(self):
        print('Loading vocabulary.')
        self.load_vocabulary()

        print('Loading data.')
        train_df, val_df = self.load_dialogues()

        x_train = train_df[self.dialogue_cols]
        x_val = val_df[self.dialogue_cols]

        # Split to lists
        self.x_train = x_train.dialogue1.values.tolist()
        self.y_train = x_train.dialogue2.values.tolist()
        self.x_val = x_val.dialogue1.values.tolist()
        self.y_val = x_val.dialogue2.values.tolist()

        self.sort_filter_tensor()
