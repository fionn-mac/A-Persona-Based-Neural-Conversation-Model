#!/usr/bin/env python

import re
import os
import sys
import random
import fileinput
import glob

from lxml import etree
from nltk import word_tokenize

train_file = open("osdb_train.txt", "w")
dev_file = open("osdb_dev.txt", "w")
vocab_file = open("vocabulary.txt", "w")

data_dir = os.getcwd()
data_set = []
word2ind = {}
vocab = []

def clean_sentence(s):
    s = re.sub(r"[^a-zA-Z.!'?]+", r" ", s)
    s = s.lower()
    words = word_tokenize(s)

    for word in words:
        if word not in word2ind:
            vocab.append(word)
            word2ind[word] = str(len(vocab))

    return ' '.join([word2ind[word] for word in words])

def parse_xml(walk_dir):
    for root, subdirs, files in os.walk(walk_dir):
        for file in files:
            xml_contents = []
            result = ''

            file_path = os.path.join(root, file)
            f = open(file_path, 'rb')

            xml_root = etree.fromstring(f.read())
            for s in xml_root:
                sentence = ''
                words = [w.text for w in s if w.text]

                for i in range(len(words)-1):
                    sentence += words[i]

                    if words[i+1] == None:
                        continue

                    last_char = words[i][-1]
                    next_char = words[i+1][0]

                    if (last_char != "'" and next_char >= 'A' and next_char <='z') or (next_char >= '0' and next_char <='9'):
                        sentence += ' '

                xml_contents.append(sentence)

            length = len(xml_contents)-2
            for i in range(0, length, 2):
                data_set.append(clean_sentence(xml_contents[i]) + ' | ' + clean_sentence(xml_contents[i+1]))

        for subdir in subdirs:
            parse_xml(os.path.join(root, subdir))

if __name__ == "__main__":
    parse_xml(os.path.join(data_dir, 'Files'))

data_size = len(data_set)
print('Total number of training samples :', data_size)

print('Writing Dialogues')
# Taking an 80% split for training
train_size = int(data_size*0.8)
random.shuffle(data_set)

for i, convo in enumerate(data_set):
    if i < train_size:
        train_file.write(convo + '\n')
    else:
        dev_file.write(convo + '\n')

print('Writing Vocabulary')

for word in vocab:
    vocab_file.write(word + '\n')
