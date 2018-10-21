from os import path
from io import open
import unicodedata
import re

import torch

import numpy as np

class Data_Preprocess(object):
    def __init__(self, dir_path, min_length=5, max_length=20):
        self.dir_path = dir_path
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.min_length = min_length
        self.max_length = max_length
        self.vocab = set(["<PAD>", "<SOS>", "<EOS>"])
        self.word2index = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>"]
        self.vocab_size = 3
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.train_lengths = []
        self.val_lengths = []
        self.run()

    def load_vocabulary(self):
        with open(path.join(self.dir_path, 'vocabulary.txt'), encoding='utf-8') as f:
            for word in f:
                word = word.strip('\n')
                self.vocab.add(word)
                self.index2word.append(word)
                self.word2index[word] = self.vocab_size
                self.vocab_size += 1

    def load_dialogues(self):
        # Load training and test set
        train_path = path.join(self.dir_path, 'train.txt')
        val_path = path.join(self.dir_path, 'val.txt')

        train_seq = [[], []]
        val_seq = [[], []]
        lengths = []

        # Iterate over dialogues of both training and test datasets
        for datafile, datalist in zip([train_path, val_path], [train_seq, val_seq]):
            with open(datafile) as file:
                lines = file.readlines()

                for line in lines:
                    line = line.split('|')

                    dialogue = line[0].split()
                    response = line[1].split()
                    len_x = len(dialogue)
                    len_y = len(response)

                    if len_x <= self.min_length or len_x >= self.max_length or \
                       len_y <= self.min_length or len_y >= self.max_length:
                       continue

                    ''' No concept of Person ID during Pre-Training '''
                    datalist[0].append([self.SOS_token] + [int(word) + 2 for word in dialogue])
                    datalist[1].append([int(word) + 2 for word in response] + [self.EOS_token])

        return train_seq, val_seq

    def convert_to_tensor(self, pairs):
        tensor_pairs = [[], []]
        lengths = []

        for i, tup in enumerate(pairs):
            tensor_pairs[0].append(torch.LongTensor(tup[0]))
            tensor_pairs[1].append(torch.LongTensor(tup[1]))
            lengths.append(len(tensor_pairs[0][-1]))

        return tensor_pairs[0], tensor_pairs[1], lengths

    def sort_and_tensor(self):
        xy_train = sorted(zip(self.x_train, self.y_train), key=lambda tup: len(tup[0]), reverse=True)
        xy_val = sorted(zip(self.x_val, self.y_val), key=lambda tup: len(tup[0]), reverse=True)

        self.x_train, self.y_train, self.train_lengths = self.convert_to_tensor(xy_train)
        self.x_val, self.y_val, self.val_lengths = self.convert_to_tensor(xy_val)

    def run(self):
        print('Loading vocabulary.')
        self.load_vocabulary()

        print('Loading data.')
        train_seq, val_seq = self.load_dialogues()

        # Split to separate lists.
        self.x_train = train_seq[0]
        self.y_train = train_seq[1]
        self.x_val = val_seq[0]
        self.y_val = val_seq[1]

        self.sort_and_tensor()
