import time
import random

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from nltk import bleu_score

from helper import Helper

class Run_Iterations(object):
    def __init__(self, model, train_in_seq, train_out_seq, train_input_lengths,
                 index2word, batch_size, num_iters, learning_rate, tracking_pair=False,
                 dev_in_seq=[], dev_out_seq=[], dev_input_lengths=[], print_every=1, plot_every=1):
        self.use_cuda = torch.cuda.is_available()
        self.model = model
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.tracking_pair = tracking_pair
        self.print_every = print_every
        self.plot_every = plot_every

        self.index2word = index2word
        ''' Lists that will contain data in the form of tensors. '''
        # Training data.
        self.train_in_seq = train_in_seq
        self.train_out_seq = train_out_seq
        self.train_input_lengths = train_input_lengths
        self.train_samples = len(self.train_in_seq)

        # Development data.
        self.dev_in_seq = dev_in_seq
        self.dev_out_seq = dev_out_seq
        self.dev_input_lengths = dev_input_lengths
        self.dev_samples = len(self.dev_in_seq)

        self.help_fn = Helper()

    def train_iters(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every self.print_every
        plot_loss_total = 0  # Reset every self.plot_every

        encoder_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.encoder.parameters()))
        decoder_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.decoder.parameters()))

        encoder_optimizer = optim.RMSprop(encoder_trainable_parameters, lr=self.learning_rate)
        decoder_optimizer = optim.RMSprop(decoder_trainable_parameters, lr=self.learning_rate)

        print('Beginning Model Training.')

        for epoch in range(1, self.num_iters + 1):
            for i in range(0, self.train_samples, self.batch_size):
                input_variables = self.train_in_seq[i : i + self.batch_size] # Batch Size x Sequence Length
                target_variables = self.train_out_seq[i : i + self.batch_size]
                lengths = self.train_input_lengths[i : i + self.batch_size]

                loss = self.model.train(input_variables, target_variables, lengths,
                                        self.criterion, encoder_optimizer, decoder_optimizer)
                print_loss_total += loss
                plot_loss_total += loss

            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.help_fn.time_slice(start, epoch / self.num_iters),
                                             epoch, epoch / self.num_iters * 100, print_loss_avg))

            if epoch % self.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # self.help_fn.show_plot(plot_losses)

    def evaluate(self, in_seq, out_seq, input_lengths):
        loss, output_words, attentions = self.model.train(in_seq, out_seq, input_lengths,
                                                          self.criterion, evaluate=True)
        return loss, output_words, attentions

    def evaluate_specific(self, in_seq, out_seq, in_len, name='tracking_pair'):
        dialogue = [self.index2word[j] for j in in_seq]
        response = [self.index2word[j] for j in out_seq]
        print('>', dialogue)
        print('=', response)

        _, output_words, attentions = self.evaluate([in_seq], [out_seq], [in_len])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)

        print('BLEU Score', bleu_score.corpus_bleu([output_sentence], [response]))
        self.help_fn.show_attention(dialogue, output_words, attentions, name=name)

    def evaluate_randomly(self, n=10):
        for i in range(n):
            ind = random.randrange(self.dev_samples)
            self.evaluate_specific(train_network, self.dev_in_seq[ind], self.dev_out_seq[ind], name=str(i))
