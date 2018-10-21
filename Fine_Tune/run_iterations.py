import time
import random

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from nltk import bleu_score

import Pre_Train
from Pre_Train.run_iterations import Run_Iterations as Base_Class
from helper import Helper

class Run_Iterations(Base_Class):
    def __init__(self, model, train_in_seq, train_out_seq, train_lengths, train_speakers, train_addressees,
                 index2word, batch_size, num_iters, learning_rate, fold_size=500000, track_minor=True,
                 tracking_pair=False, val_in_seq=[], val_out_seq=[], val_lengths=[], val_speakers=[],
                 val_addressees=[], print_every=1, plot_every=1):
        Base_Class.__init__(self, model, train_in_seq, train_out_seq, train_lengths, index2word,
                     batch_size, num_iters, learning_rate, fold_size, track_minor, tracking_pair,
                     val_in_seq, val_out_seq, val_lengths, print_every, plot_every)
        '''
        in_seq and out_seq = list of tensors
        lengths = list
        speakers, addressees = tensors
        '''
        # Training data.
        self.train_speakers = train_speakers
        self.train_addressees = train_addressees

        # valelopment data.
        self.val_speakers = val_speakers
        self.val_addressees = val_addressees

    def train_iters(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every self.print_every
        plot_loss_total = 0  # Reset every self.plot_every

        encoder_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.encoder.parameters()))
        decoder_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.decoder.parameters()))

        if self.tracking_pair:
            ind = random.randint(0, self.train_samples)
            self.tracking_pair = [self.train_in_seq[ind], self.train_out_seq[ind], self.train_in_seq[ind].size()[0],
                                  [self.train_speakers[ind:ind+1], self.train_addressees[ind:ind+1]]]
            if self.use_cuda:
                self.tracking_pair[0] = self.tracking_pair[0].cuda()
                self.tracking_pair[1] = self.tracking_pair[1].cuda()
                self.tracking_pair[3][0] = self.tracking_pair[3][0].cuda()
                self.tracking_pair[3][1] = self.tracking_pair[3][1].cuda()

        in_folds = []
        out_folds = []
        fold_lengths = []
        fold_speakers = []
        fold_addressees = []

        for i in range(0, self.train_samples, self.fold_size):
            in_folds.append(self.train_in_seq[i : i + self.fold_size])
            out_folds.append(self.train_out_seq[i : i + self.fold_size])
            fold_lengths.append(self.train_lengths[i : i + self.fold_size])
            fold_speakers.append(self.train_speakers[i : i + self.fold_size])
            fold_addressees.append(self.train_addressees[i : i + self.fold_size])

        self.train_in_seq = in_folds
        self.train_out_seq = out_folds
        self.train_lengths = fold_lengths
        self.train_speakers = fold_speakers
        self.train_addressees = fold_addressees

        del in_folds, out_folds, fold_lengths, fold_speakers, fold_addressees

        print('Number of folds               :', len(self.train_in_seq), '\n')
        print('Beginning Model Training.')

        fold_number = 1 # Keep track of fold over which model is training.
        inputs = zip(self.train_in_seq, self.train_out_seq, self.train_lengths, \
                     self.train_speakers, self.train_addressees)

        for in_fold, out_fold, fold_lengths, fold_speakers, fold_addressees in inputs:
            # Initialze Optimizers
            learning_rate = self.learning_rate
            encoder_optimizer = optim.Adam(encoder_trainable_parameters, lr=learning_rate)
            decoder_optimizer = optim.Adam(decoder_trainable_parameters, lr=learning_rate)

            # Convert fold contents to cuda
            if self.use_cuda:
                in_fold = self.help_fn.to_cuda(in_fold)
                out_fold = self.help_fn.to_cuda(out_fold)
                fold_speakers = fold_speakers.cuda()
                fold_addressees = fold_addressees.cuda()

            fold_size = len(in_fold)
            fraction = fold_size // 10

            print('Starting Fold                 :', fold_number)

            for epoch in range(1, self.num_iters + 1):
                for i in range(0, fold_size, self.batch_size):
                    input_variables = in_fold[i : i + self.batch_size] # Batch Size x Sequence Length
                    target_variables = out_fold[i : i + self.batch_size]
                    lengths = fold_lengths[i : i + self.batch_size]
                    addressees = fold_addressees[i : i + self.batch_size].view(1, -1) # 1 x Batch Size

                    loss = self.model.train(input_variables, target_variables, lengths, self.criterion,
                                            encoder_optimizer, decoder_optimizer, people=addressees)

                    print_loss_total += loss
                    plot_loss_total += loss

                    if self.track_minor and i > 0 and (i - self.batch_size) // fraction < i // fraction:
                        now = time.time()
                        print('Completed %.4f Percent of Epoch %d in %s Minutes' % ((i + self.batch_size) / fold_size * 100,
                                                                                    epoch, self.help_fn.as_minutes(now - start)))

                if self.tracking_pair: self.evaluate_specific(*self.tracking_pair)

                if epoch % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (self.help_fn.time_slice(start, epoch / self.num_iters),
                                                 epoch, epoch / self.num_iters * 100, print_loss_avg))

                if epoch % self.plot_every == 0:
                    plot_loss_avg = plot_loss_total / self.plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                if epoch % 5 == 0:
                    self.learning_rate = max(self.learning_rate * 0.80, 0.001)
                    encoder_optimizer = optim.Adam(encoder_trainable_parameters, lr=self.learning_rate)
                    decoder_optimizer = optim.Adam(decoder_trainable_parameters, lr=self.learning_rate)

            # Convert fold contents back to cpu
            if self.use_cuda:
                in_fold = self.help_fn.to_cpu(in_fold)
                out_fold = self.help_fn.to_cpu(out_fold)
                fold_speakers = fold_speakers.cpu()
                fold_addressees = fold_addressees.cpu()

            fold_number += 1
            # self.help_fn.show_plot(plot_losses)

    def evaluate(self, in_seq, out_seq, lengths, people):
        loss, output_words, attentions = self.model.evaluate(in_seq, out_seq, lengths, self.criterion, people=people)
        return loss, output_words, attentions

    def evaluate_specific(self, in_seq, out_seq, in_len, person, name='tracking_pair'):
        dialogue = [self.index2word[j] for j in in_seq]
        response = [self.index2word[j] for j in out_seq]
        print('>', dialogue, 'By :', person[0].item())
        print('=', response, 'By :', person[1].item())

        _, output_words, attentions = self.evaluate([in_seq], [out_seq], [in_len], person[0].view(1, 1))
        try:
            target_index = output_words[0].index('<EOS>') + 1
        except ValueError:
            target_index = len(output_words[0])

        output_words = output_words[0][:target_index]
        attentions = attentions[0, :target_index, :].view(target_index, -1)

        output_sentence = ' '.join(output_words)
        print('<', output_sentence)

        print('BLEU Score', bleu_score.corpus_bleu([output_sentence], [response]))
        self.help_fn.show_attention(dialogue, output_words, attentions, name=name)

    def evaluate_randomly(self, n=10):
        if self.use_cuda:
            self.val_in_seq = self.help_fn.to_cuda(self.val_in_seq)
            self.val_out_seq = self.help_fn.to_cuda(self.val_out_seq)
            self.val_speakers = self.val_speakers.cuda()
            self.val_addressees = self.val_addressees.cuda()

        print('Evaluating Model on %d random validation samples' % (n))

        for i in range(n):
            ind = random.randrange(self.val_samples)
            self.evaluate_specific(self.val_in_seq[ind], self.val_out_seq[ind], len(self.val_in_seq[ind]),
                                   [self.val_speakers[ind:ind+1], self.val_addressees[ind:ind+1]], name=str(i))
