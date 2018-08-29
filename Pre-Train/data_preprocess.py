from os import path
from io import open
import unicodedata
import re

import torch

import numpy as np
import pandas as pd

class Data_Preprocess(object):
    def __init__(self, path, max_length=10):
        self.path = path
        self.PAD_token = 0
        self.EOS_token = 2
        self.vocab = set("<EOS>")
        self.word2index = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>"]
        self.word2count = {}
        self.vocab_size = 3
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.lengths_train = []
        self.lengths_val = []
        self.dialogue_cols = ['dialogue1', 'dialogue2']
        self.max_length = max_length
        self.use_cuda = torch.cuda.is_available()
        self.run()

    def load_vocabulary(self):
        with open(self.path + 'vocabulary.txt', encoding='utf-8') as f:
            for word in f:
                word = word.strip('\n')
                self.vocab.add(word)
                self.index2word.append(word)
                self.word2index[word] = self.vocab_size
                self.vocab_size += 1

    def load_dialogues(self):
        # Load training and test set
        train_df = pd.read_csv(path.join(self.path, 'osdb_train_mini.txt'), sep='|')
        val_df = pd.read_csv(path.join(self.path, 'osdb_dev_mini.txt'), sep='|')

        lengths = []

        # Iterate over dialogues of both training and test datasets
        for dataset in [train_df, val_df]:
            for index, row in dataset.iterrows():
                # Iterate through the text of both the dialogues of the row
                for dialogue in self.dialogue_cols:
                    d2n = []  # Dialogue Numbers Representation
                    for i, word_id in enumerate(row[dialogue].split()):
                        ''' No concept of Person ID during Pre-Training '''
                        ''' Considering only first |max_length-1| words, with final word being EOS token '''

                        if len(d2n) < self.max_length - 1:
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

    def sort_by_lengths(self):
        xy_train = sorted(zip(self.x_train, self.y_train), key=lambda tup: len(tup[0]), reverse=True)
        xy_val = sorted(zip(self.x_val, self.y_val), key=lambda tup: len(tup[0]), reverse=True)

        for i, tup in enumerate(xy_train):
            self.x_train[i] = torch.LongTensor(tup[0])
            self.y_train[i] = torch.LongTensor(tup[1])
            if self.use_cuda:
                self.x_train[i] = self.x_train[i].cuda()
                self.y_train[i] = self.y_train[i].cuda()

            self.lengths_train.append(len(self.x_train[i]))

        for i, tup in enumerate(xy_val):
            self.x_val[i] = torch.LongTensor(tup[0])
            self.y_val[i] = torch.LongTensor(tup[1])
            if self.use_cuda:
                self.x_val[i] = self.x_val[i].cuda()
                self.y_val[i] = self.y_val[i].cuda()

            self.lengths_val.append(len(self.x_val[i]))

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

        self.sort_by_lengths()
