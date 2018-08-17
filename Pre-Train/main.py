import time
import random

import torch
import torch.nn as nn
from torch import optim

from dataPreprocess import DataPreprocess
from embeddingGoogle import GetEmbedding
from encoderRNN import EncoderRNN
from decoderRNN import DecoderRNN
from trainNetwork import TrainNetwork
from helper import Helper

use_cuda = torch.cuda.is_available()

def trainIters(model, in_seq, out_seq, input_lengths, max_length, batch_size=1,
               n_iters=5, learning_rate=0.001, print_every=1, plot_every=1):

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
            input_variables = in_seq[i : i + batch_size].permute(1, 0) # Sequence Length x Batch Size
            target_variables = out_seq[i : i + batch_size].permute(1, 0)
            lengths = input_lengths[i : i + batch_size]

            loss = model.train(input_variables, target_variables, lengths,
                               encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        evaluateRandomly(train_network, in_seq, out_seq, lengths, index2word, 1)

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

def evaluate(train_network, sentence):
    output_words, attentions = train_network.evaluate(sentence)
    return output_words, attentions

def evaluateRandomly(train_network, in_seq, out_seq, lengths, index2word, n=10):
    samples = len(in_seq)
    for i in range(n):
        ind = random.randrange(samples)
        print('>', [index2word[j] for j in in_seq[ind]])
        print('=', [index2word[j] for j in out_seq[ind]])
        output_words, attentions = evaluate(train_network, in_seq[ind:ind+1].permute(1, 0))
        print('<', output_words)
        print('')
        # helpFn.showAttention([index2word[j] for j in in_seq[ind]], output_words, attentions)

if __name__ == "__main__":

    hidden_size = 1024
    batch_size = 32
    num_layers = 3
    max_length = 20

    data_preprocess = DataPreprocess("./Datasets/OpenSubtitles/", max_length=max_length)
    in_seq = data_preprocess.x_train
    out_seq = data_preprocess.y_train
    lengths = data_preprocess.lengths_train
    word2index = data_preprocess.word2index
    index2word = data_preprocess.index2word
    word2count = data_preprocess.word2count
    vocab_size = data_preprocess.vocab_size

    helpFn = Helper(max_length)

    ''' Use pre-trained word embeddings '''
    embedding = GetEmbedding(word2index, word2count, "../Embeddings/GoogleNews/")

    encoder = EncoderRNN(hidden_size, embedding.embedding_matrix, batch_size=batch_size,
                         num_layers=num_layers, use_embedding=True, train_embedding=False)
    decoder = DecoderRNN(hidden_size, embedding.embedding_matrix, num_layers=num_layers,
                         use_embedding=True, train_embedding=False, dropout_p=0.1)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Training Network.")
    print("Input data dimensions:", in_seq.size())

    train_network = TrainNetwork(encoder, decoder, index2word, max_length, batch_size=batch_size, num_layers=num_layers)
    trainIters(train_network, in_seq, out_seq, lengths, max_length, batch_size=batch_size)

    evaluateRandomly(train_network, in_seq, out_seq, lengths, index2word)
