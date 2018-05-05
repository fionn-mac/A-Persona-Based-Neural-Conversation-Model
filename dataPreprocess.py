from io import open
import unicodedata
import re

import torch

import numpy as np
import pandas as pd

class DataPreprocess(object):
    def __init__(self, path, max_length=10):
        self.path = path
        self.vocab = set("<EOS>")
        self.people = set()
        self.word2ind = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>"]
        self.vocab_size = 3
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.lengths_train = []
        self.lengths_val = []
        self.speaker_list_train = list()
        self.addressee_list_train = list()
        self.speaker_list_val = list()
        self.addressee_list_val = list()
        self.dialogue_cols = ['dialogue1', 'dialogue2']
        self.max_length = max_length
        self.use_cuda = torch.cuda.is_available()
        self.run()

    def load_vocabulary(self):
        with open(self.path + 'movie_25000') as f:
            for word in f:
                if word.strip('\n') not in self.vocab:
                    self.vocab.add(word.strip('\n'))
                    self.index2word.append(word.strip('\n'))
                    self.word2ind[word.strip('\n')] = self.vocab_size
                    self.vocab_size += 1

    def load_dialogues(self):
        # Load training and test set
        train_df = pd.read_csv(self.path + 'speaker_addressee_train.txt', sep='|')
        val_df = pd.read_csv(self.path + 'speaker_addressee_dev.txt', sep='|')

        train = [[], []]
        val = [[], []]
        lengths = []

        # Iterate over the questions only of both training and test datasets
        for dataset, data_type in zip([train_df, val_df], [train, val]):
            for index, row in dataset.iterrows():
                # Iterate through the text of both the dialogues of the row
                for dialogue, person in zip(self.dialogue_cols, data_type):
                    d2n = []  # Dialogue Numbers Representation
                    for i, word_id in enumerate(row[dialogue].split(' ')):
                        if i == 0: person.append(int(word_id))
                        elif len(d2n) < self.max_length - 1: d2n.append(int(word_id) + 2)

                    # Replace questions as word to question as number representation
                    dataset.set_value(index, dialogue, d2n + [2])

        return train_df, train[0], train[1], val_df, val[0], val[1]

    def sort_by_lengths(self):
        xysa_train = sorted(zip(self.x_train, self.y_train, self.speaker_list_train, self.addressee_list_train),
                            key=lambda tup: len(tup[0]), reverse=True)
        xysa_val = sorted(zip(self.x_val, self.y_val, self.speaker_list_val, self.addressee_list_val),
                          key=lambda tup: len(tup[0]), reverse=True)

        for i, tup in enumerate(xysa_train):
            self.x_train[i] = tup[0]
            self.y_train[i] = tup[1]
            self.speaker_list_train[i] = tup[2]
            self.addressee_list_train[i] = tup[3]
            self.lengths_train.append(len(tup[0]))

        for i, tup in enumerate(xysa_val):
            self.x_val[i] = tup[0]
            self.y_val[i] = tup[1]
            self.speaker_list_val[i] = tup[2]
            self.addressee_list_val[i] = tup[3]
            self.lengths_val.append(len(tup[0]))

    def get_people(self):
        people = self.speaker_list_train + self.speaker_list_val +\
                 self.addressee_list_train + self.addressee_list_val

        for person in people:
            if person not in self.people:
                self.people.add(person)

    def run(self):
        print('Loading vocabulary.')
        self.load_vocabulary()

        print('Loading data.')
        train_df, self.speaker_list_train, self.addressee_list_train, val_df, self.speaker_list_val, self.addressee_list_val = self.load_dialogues()

        x_train = train_df[self.dialogue_cols]
        x_val = val_df[self.dialogue_cols]

        # Split to lists
        self.x_train = x_train.dialogue1.values.tolist()
        self.y_train = x_train.dialogue2.values.tolist()
        self.x_val = x_val.dialogue1.values.tolist()
        self.y_val = x_val.dialogue2.values.tolist()

        self.sort_by_lengths()

        # self.x_train = torch.LongTensor(self.x_train)
        # self.y_train = torch.LongTensor(self.y_train)
        # self.x_val = torch.LongTensor(self.x_val)
        # self.y_val = torch.LongTensor(self.y_val)
        # 
        # if self.use_cuda:
        #     self.x_train = self.x_train.cuda()
        #     self.y_train = self.y_train.cuda()
        #     self.x_val = self.x_val.cuda()
        #     self.y_val = self.y_val.cuda()

        self.get_people()
