import time
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from dataPreprocess import DataPreprocess
from embeddingGoogle import GetEmbedding
from encoderRNN import EncoderRNN
from decoderRNN import DecoderRNN
from trainNetwork import TrainNetwork
from helper import Helper

use_cuda = torch.cuda.is_available()

def trainIters(model, in_seq, out_seq, people, input_lengths, max_length, batch_size=1,
               n_iters=75, learning_rate=0.01, print_every=1, plot_every=1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                      lr=learning_rate)
    decoder_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                      lr=learning_rate)

    samples = len(in_seq)
    samples -= samples % batch_size
    criterion = nn.NLLLoss(ignore_index=0)

    for epoch in range(1, n_iters + 1):
        for i in range(0, samples, batch_size):
            input_variables = torch.cuda.LongTensor(in_seq[i : i + batch_size]).permute(1, 0) # Sequence Length x Batch Size
            target_variables = Variable(torch.cuda.LongTensor(out_seq[i : i + batch_size]).permute(1, 0))
            lengths = input_lengths[i : i + batch_size]
            speaker = torch.cuda.LongTensor(people[1][i : i + batch_size]).view(1, 1)

            loss = model.train(input_variables, target_variables, speaker, lengths,
                               encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (helpFn.time_slice(start, epoch / n_iters),
                                         epoch, epoch / n_iters * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # helpFn.show_plot(plot_losses)

def evaluate(train_network, sentence, people):
    output_words, attentions = train_network.evaluate(sentence, people)
    return output_words, attentions

def evaluateRandomly(train_network, in_seq, out_seq, people, lengths, index2word, n=10):
    samples = len(in_seq)
    for i in range(n):
        ind = random.randrange(samples)
        print('>', [index2word[j] for j in in_seq[ind]])
        print('=', [index2word[j] for j in out_seq[ind]])
        output_words, attentions = evaluate(train_network, torch.cuda.LongTensor([in_seq[ind]]).permute(1, 0),
                                            torch.cuda.LongTensor([people[1][ind]]).view(1, 1))
        print('<', output_words)
        print('')
        helpFn.showAttention([index2word[j] for j in in_seq[ind]], output_words, attentions)

if __name__ == "__main__":

    hidden_size = 1024
    batch_size = 1

    data_preprocess = DataPreprocess("./Datasets/Neural-Dialogue-Generation/data/")
    max_length = data_preprocess.max_length
    in_seq = data_preprocess.x_train
    out_seq = data_preprocess.y_train
    lengths = data_preprocess.lengths_train
    speakers = data_preprocess.speaker_list_train
    addressees = data_preprocess.addressee_list_train
    index2word = data_preprocess.index2word
    vocab_size = data_preprocess.vocab_size
    personas = len(data_preprocess.people) + 1

    helpFn = Helper(max_length)

    encoder = EncoderRNN(hidden_size, (vocab_size, 300), batch_size)
    decoder = DecoderRNN(hidden_size, (vocab_size, 300), (personas, 300))

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Training Network.")
    train_network = TrainNetwork(encoder, decoder, index2word, max_length, batch_size)
    trainIters(train_network, in_seq, out_seq, [speakers, addressees], lengths, max_length, batch_size)

    evaluateRandomly(train_network, in_seq, out_seq, [speakers, addressees], lengths, index2word)
