from os import path
from io import open
import unicodedata

import torch

import numpy as np

import Pre_Train
from Pre_Train.data_preprocess import Data_Preprocess as Base_Class

class Data_Preprocess(Base_Class):
    def __init__(self, dir_path, min_length=5, max_length=20):
        self.train_speakers = list()
        self.train_addressees = list()
        self.val_speakers = list()
        self.val_addressees = list()
        self.people = set()
        Base_Class.__init__(self, dir_path, min_length, max_length)

    def load_dialogues(self):
        # Load training and test set
        train_path = path.join(self.path, 'train.txt')
        val_path = path.join(self.path, 'val.txt')

        train_seq = [[], []]
        val_seq = [[], []]

        train_info = [train_seq, self.train_speakers, self.train_addressees]
        val_info = [val_seq, self.val_speakers, self.val_addressees]

        # Iterate over dialogues of both training and test datasets
        for datafile, datalist in zip([train_path, val_path], [train_info, val_info]):
            with open(datafile) as file:
                lines = file.readlines()

                for line in lines:
                    line = line.split('|')

                    input_1 = line[0].split()
                    speaker = input_1[0]
                    dialogue = input_1[1:]

                    input_2 = line[1].split()
                    addressee = input_2[0]
                    response = input_2[1:]

                    len_x = len(dialogue)
                    len_y = len(response)

                    if len_x <= self.min_length or len_x >= self.max_length or \
                       len_y <= self.min_length or len_y >= self.max_length:
                       continue

                    ''' First number in each list corresponds to Person ID of speaker '''
                    datalist[0][0].append([self.SOS_token] + [int(word) + 2 for word in dialogue])
                    datalist[0][1].append([int(word) + 2 for word in response] + [self.EOS_token])
                    datalist[1].append(speaker)
                    datalist[2].append(addressee)

        return train_seq, val_seq

    def get_people(self):
        people = self.train_speakers + self.val_speakers +\
                 self.train_addressees + self.val_addressees

        for person in people:
            if person not in self.people:
                self.people.add(person)

    def filter_and_tensor(self, quadruplets):
        tensor_pairs = [[], []]
        lengths = []
        speakers = []
        addressees = []

        for i, tup in enumerate(pairs):
            tensor_pairs[0].append(torch.LongTensor(tup[0]))
            tensor_pairs[1].append(torch.LongTensor(tup[1]))
            lengths.append(len(tensor_pairs[0][-1]))

            speakers.append(tup[2])
            addressees.append(tup[3])

        speakers = torch.LongTensor(speakers)
        addressees = torch.LongTensor(addressees)

        return cleaned_pairs[0], cleaned_pairs[1], lengths, speakers, addressees

    def sort_and_tensor(self):
        xysa_train = sorted(zip(self.x_train, self.y_train, self.train_speakers, self.train_addressees), key=lambda tup: len(tup[0]), reverse=True)
        xysa_val = sorted(zip(self.x_val, self.y_val, self.val_speakers, self.val_addressees), key=lambda tup: len(tup[0]), reverse=True)

        self.x_train, self.y_train, self.train_lengths, self.train_speakers, self.train_addressees = self.convert_to_tensor(xysa_train)
        self.x_val, self.y_val, self.val_lengths, self.val_speakers, self.val_addressees = self.convert_to_tensor(xysa_val)

    def run(self):
        print('Loading vocabulary.')
        self.load_vocabulary()

        print('Loading data.')
        train_seq, val_seq = self.load_dialogues()

        # Split to lists
        self.x_train = train_seq[0]
        self.y_train = train_seq[1]
        self.x_val = val_seq[0]
        self.y_val = val_seq[1]

        self.get_people()
        self.sort_and_tensor()
