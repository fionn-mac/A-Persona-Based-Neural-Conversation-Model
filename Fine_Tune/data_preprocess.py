import re

from io import open
import unicodedata

import torch

import numpy as np
import pandas as pd

import Pre_Train
from Pre_Train.data_preprocess import Data_Preprocess as Base_Class

class Data_Preprocess(Base_Class):
    def __init__(self, path, min_length=5, max_length=20):
        self.val_speakers = list()
        self.val_addressees = list()
        self.people = set()
        Base_Class.__init__(self, path, min_length, max_length)

    def load_dialogues(self):
        # Load training and test set
        train_df = pd.read_csv(self.path + 'friends_train.txt', sep='|')
        val_df = pd.read_csv(self.path + 'friends_dev.txt', sep='|')

        '''
        train contains the speakers and addressees of training split.
        val contains the speakers and addressees of validation split.
        '''
        train = [[], []]
        val = [[], []]
        lengths = []

        # Iterate over dialogues and characters of both training and test datasets
        for dataset, data_type in zip([train_df, val_df], [train, val]):
            for index, row in dataset.iterrows():
                # Iterate through the text of both the dialogues of the row
                for dialogue, person in zip(self.dialogue_cols, data_type):
                    d2n = []  # Dialogue Numbers Representation
                    for i, word_id in enumerate(row[dialogue].split()):
                        ''' First number is person ID '''
                        if i == 0:
                            person.append(int(word_id))

                        else:
                            effective_id = int(word_id) + 2
                            word = self.index2word[effective_id]

                            d2n.append(effective_id)
                            if word not in self.word2count:
                                self.word2count[word] = 0
                            self.word2count[word] += 1

                    # Replace |questions as word| to |question as number| representation
                    # Add <EOS> token at end of dialogue.
                    dataset.at[index, dialogue] = d2n + [self.EOS_token]

        return train_df, train[0], train[1], val_df, val[0], val[1]

    def get_people(self):
        people = self.train_speakers + self.val_speakers +\
                 self.train_addressees + self.val_addressees

        for person in people:
            if person not in self.people:
                self.people.add(person)

    def len_check(self, x, y):
        len_x = len(x)
        len_y = len(y)

        if len_x >= self.min_length and len_x <= self.max_length and \
           len_y >= self.min_length and len_y <= self.max_length:
           return True

        return False

    def filter_and_tensor(self, quadruplets):
        cleaned_pairs = [[], []]
        lengths = []
        speakers = []
        addressees = []

        for i, tup in enumerate(quadruplets):
            if not self.len_check(tup[0], tup[1]):
                continue

            cleaned_pairs[0].append(torch.LongTensor(tup[0]))
            cleaned_pairs[1].append(torch.LongTensor(tup[1]))

            lengths.append(len(cleaned_pairs[0][-1]))
            speakers.append(tup[2])
            addressees.append(tup[3])

        speakers = torch.LongTensor(speakers)
        addressees = torch.LongTensor(addressees)

        return cleaned_pairs[0], cleaned_pairs[1], lengths, speakers, addressees

    def sort_filter_tensor(self):
        xysa_train = sorted(zip(self.x_train, self.y_train, self.train_speakers, self.train_addressees), key=lambda tup: len(tup[0]), reverse=True)
        xysa_val = sorted(zip(self.x_val, self.y_val, self.val_speakers, self.val_addressees), key=lambda tup: len(tup[0]), reverse=True)

        self.x_train, self.y_train, self.train_lengths, self.train_speakers, self.train_addressees = self.filter_and_tensor(xysa_train)
        self.x_val, self.y_val, self.val_lengths, self.val_speakers, self.val_addressees = self.filter_and_tensor(xysa_val)

    def run(self):
        print('Loading vocabulary.')
        self.load_vocabulary()

        print('Loading data.')
        train_df, self.train_speakers, self.train_addressees, val_df, self.val_speakers, self.val_addressees = self.load_dialogues()

        x_train = train_df[self.dialogue_cols]
        x_val = val_df[self.dialogue_cols]

        # Split to lists
        self.x_train = x_train.dialogue1.values.tolist()
        self.y_train = x_train.dialogue2.values.tolist()
        self.x_val = x_val.dialogue1.values.tolist()
        self.y_val = x_val.dialogue2.values.tolist()

        self.get_people()
        self.sort_filter_tensor()
