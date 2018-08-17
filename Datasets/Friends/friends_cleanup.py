import re
import random
import string

from nltk import word_tokenize

data = []
printable = set(string.printable)
vocab = {}
vocab2 = []
actors = {"ross" : 1,  "monica" : 2, "joey" : 3, "rachel" : 4, "chandler" : 5, "phoebe" : 6, "cameo" : 7}
actors2 = ["ross",  "monica", "joey", "rachel", "chandler", "phoebe", "cameo"]

train_file = open("friends_train.txt", "w")
dev_file = open("friends_dev.txt", "w")
vocab_file = open("vocabulary.txt", "w")
cast_file = open("cast.txt", "w")

train_file.write('dialogue1|dialogue2\n')
dev_file.write('dialogue1|dialogue2\n')

with open('dialogue.csv') as f:
    lines =  f.readlines()

    data_size = len(lines)

    for k in range(1, data_size, 2):
        if k+1 >= data_size:
            break

        dialogue = lines[k].split('|')[-3:]
        response = lines[k+1].split('|')[-3:]

        if dialogue[0] == response[0]:
            temp = [*dialogue[1:], *response[1:]]
            for i, val in enumerate(temp):
                val = re.sub('\[.*\]|\(.*\)|\n|"', '', val)
                val = ' '.join(re.findall(r"[\w']+|[.,!?;]", val))

                temp[i] = ''
                for j in val:
                    if j in printable: temp[i] += j

                if i%2:
                    temp2 = word_tokenize(temp[i])
                    for word in temp2:
                        if word not in vocab:
                            vocab[word] = str(len(vocab2)+1)
                            vocab2.append(word)

                    temp[i] = ' '.join([vocab[word] for word in temp2])

                else:
                    temp2 = temp[i].split()[0]
                    if temp2 not in actors:
                        temp2 = "cameo"

                    temp[i] = str(actors[temp2])

            if (len(temp[1]) > 1 and len(temp[3]) > 1):
                data_line = temp[0] + ' ' + temp[1] + '|' + temp[2] + ' ' + temp[3]
                data.append(data_line)

        else:
            k -= 1

print('Dialogues')
random.shuffle(data)
for i, convo in enumerate(data):
    if i < 27308:
        train_file.write(convo + '\n')
    else:
        dev_file.write(convo + '\n')

print('Vocabulary')

for word in vocab2:
    vocab_file.write(word + '\n')

print('Actors')

for word in actors2:
    cast_file.write(word + '\n')
