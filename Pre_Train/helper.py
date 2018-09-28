import time
import math

import torch
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Helper(object):
    def __init__(self):
        pass

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

    def to_cuda(self, tensors):
        for i, tensor in enumerate(tensors):
            tensors[i] = tensor.cuda()

        return tensors

    def to_cpu(self, tensors):
        for i, tensor in enumerate(tensors):
            tensors[i] = tensor.cpu()

        return tensors

    def show_plot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    def show_attention(self, input_sentence, output_words, attentions, show=False, name=None):
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

        if show: plt.show()
        if name: plt.savefig(str(name) + '.png')
