import time
import math

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Helper(object):
    def __init__(self, max_length, PAD_token=0):
        self.EOS_token = 2
        self.PAD_token = PAD_token
        self.max_length = max_length
        self.use_cuda = torch.cuda.is_available()

    def as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_slice(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.as_minutes(s), self.as_minutes(rs))

    def show_plot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    def indexes_from_sentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence]

    def variable_from_sentence(self, lang, sentence):
        indexes = self.indexes_from_sentence(lang, sentence) + [self.EOS_token]
        length = len(indexes)
        indexes += [self.PAD_token]*(self.max_length - length)
        result = Variable(torch.LongTensor(indexes)).view(-1, 1)

        if self.use_cuda:
            return result.cuda(), length
        else:
            return result, length

    def variables_from_pair(self, input_lang, output_lang, pair):
        input_variable, input_length = self.variable_from_sentence(input_lang, pair[0])
        target_variable, target_length = self.variable_from_sentence(output_lang, pair[1])
        return input_variable, input_length, target_variable, target_length

    def showAttention(self, input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
