import re
import random
import string
import unicodedata

from nltk import word_tokenize

data = []
word2ind = {}
vocab = set()
actors = {"ross" : 1,  "monica" : 2, "joey" : 3, "rachel" : 4, "chandler" : 5, "phoebe" : 6, "cameo" : 7}
ind2actor = ["ross",  "monica", "joey", "rachel", "chandler", "phoebe", "cameo"]

with open("vocabulary.txt", "r") as f:
    lines = f.readlines()

    for i, line in enumerate(lines):
        word = line.strip()
        vocab.add(word)
        word2ind[word] = str(i + 1)

train_file = open("train.txt", "w")
dev_file = open("val.txt", "w")
cast_file = open("cast.txt", "w")

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub('\[.*?\]|\(.*?\)|\n|"', '', s)
    return s

with open('dialogue.csv') as f:
    lines =  f.readlines()

    data_size = len(lines)

    for k in range(1, data_size, 2):
        if k+1 >= data_size:
            break

        dialogue = lines[k].split('|')[-3:]
        response = lines[k+1].split('|')[-3:]

        # Both dialogues from same episode
        if dialogue[0] == response[0]:
            dialogue_pair = [*dialogue[1:], *response[1:]]

            for i, val in enumerate(dialogue_pair):
                val = normalize_string(val)

                # Dialogue is at even indices in dialogue_pair array.
                if i%2:
                    # Tokenize response string.
                    val = word_tokenize(val)
                    for j, word in enumerate(val):
                        if word not in vocab:
                            val[j] = 'UNK'

                    dialogue_pair[i] = ' '.join([word2ind[word] for word in val])

                # Speaker name is at odd indices in dialogue_pair array.
                else:
                    if val not in actors:
                        val = "cameo"

                    dialogue_pair[i] = str(actors[val])

            # Check both dialogue and reponse are non-empty
            if (len(dialogue_pair[1]) > 1 and len(dialogue_pair[3]) > 1):
                data_line = dialogue_pair[0] + ' ' + dialogue_pair[1] + '|' + dialogue_pair[2] + ' ' + dialogue_pair[3]
                data.append(data_line)

        # Next line lies in new dialogue; decrement index by 1.
        else:
            k -= 1

data_size = len(data)
print('Total number of training samples :', data_size)

print('Writing Dialogues')
# Taking an 80% split for training
train_size = int(data_size*0.8)
random.shuffle(data)

for i, convo in enumerate(data):
    if i < train_size:
        train_file.write(convo + '\n')
    else:
        dev_file.write(convo + '\n')

print('Writing Actors')

for word in ind2actor:
    cast_file.write(word + '\n')
