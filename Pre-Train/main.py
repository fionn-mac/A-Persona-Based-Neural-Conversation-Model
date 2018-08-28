import time
import random

import torch
import torch.nn as nn
from torch import optim

from nltk import bleu_score

from data_preprocess import Data_Preprocess
from embedding_google import Get_Embedding
from encoder_rnn import Encoder_RNN
from decoder_rnn import Decoder_RNN
from train_network import Train_Network
from helper import Helper

use_cuda = torch.cuda.is_available()
helpFn = Helper()

def trainIters(model, in_seq, out_seq, input_lengths, index2word, batch_size=1, n_iters=10,
               learning_rate=0.001, print_every=1, plot_every=1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_trainable_parameters = list(filter(lambda p: p.requires_grad, model.encoder.parameters()))
    decoder_trainable_parameters = list(filter(lambda p: p.requires_grad, model.decoder.parameters()))

    encoder_optimizer = optim.RMSprop(encoder_trainable_parameters, lr=learning_rate)
    decoder_optimizer = optim.RMSprop(decoder_trainable_parameters, lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=0)

    print('Beginning Model Training.')
    samples = len(in_seq)

    for epoch in range(1, n_iters + 1):
        for i in range(0, samples, batch_size):
            input_variables = in_seq[i : i + batch_size] # Batch Size x Sequence Length
            target_variables = out_seq[i : i + batch_size]
            lengths = input_lengths[i : i + batch_size]

            loss = model.train(input_variables, target_variables, lengths,
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

def evaluate(train_network, sentence):
    output_words, attentions = train_network.evaluate([sentence])
    return output_words, attentions

def evaluate_specific(train_network, in_seq, out_seq, index2word, name='tracking_pair'):
    dialogue = [index2word[j] for j in in_seq]
    response = [index2word[j] for j in out_seq]
    print('>', dialogue)
    print('=', response)

    output_words, attentions = evaluate(train_network, [in_seq])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)

    print('BLEU Score', bleu_score.corpus_bleu([output_sentence], [response]))
    helpFn.showAttention(dialogue, output_words, attentions, name=name)

def evaluate_randomly(train_network, in_seq, out_seq, index2word, n=10):
    samples = len(in_seq)
    for i in range(n):
        ind = random.randrange(samples)
        evaluate_specific(train_network, in_seq[ind], out_seq[ind], index2word, name=str(i))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--hidden_size", type=int, help="GRU Hidden State Size", default=1024)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("-l", "--max_length", type=int, help="Maximum Sentence Length.", default=20)
    parser.add_argument("--num_layers", type=int, help="Number of layers in Encoder and Decoder", default=3)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset directory.", default='../Datasets/OpenSubtitles/')
    parser.add_argument("-e", "--embedding_file", type=str, help="File containing word embeddings.", default='../../Embeddings/GoogleNews/GoogleNews-vectors-negative300.bin.gz')

    args = parser.parse_args()

    hidden_size = args.hidden_size
    batch_size = args.batch_size
    max_length = args.max_length
    num_layers = args.num_layers
    dataset = args.dataset
    embedding_file = args.embedding_file

    print('Model Parameters:')
    print('Hidden Size                :', hidden_size)
    print('Batch Size                 :', batch_size)
    print('Number of Layers           :', num_layers)
    print('Max. input length          :', max_length)
    print('--------------------------------------\n')

    print('Reading input data.')
    data_preprocess = DataPreprocess(dataset, max_length=max_length)
    in_seq = data_preprocess.x_train
    out_seq = data_preprocess.y_train
    lengths = data_preprocess.lengths_train
    word2index = data_preprocess.word2index
    index2word = data_preprocess.index2word
    word2count = data_preprocess.word2count
    vocab_size = data_preprocess.vocab_size
    print("Number of training Samples  :", len(in_seq))

    print('Creating Word Embedding.')

    ''' Use pre-trained word embeddings '''
    embedding = Get_Embedding(word2index, word2count, embedding_file)

    encoder = Encoder_RNN(hidden_size, embedding.embedding_matrix, batch_size=batch_size,
                          num_layers=num_layers, use_embedding=True, train_embedding=False)
    decoder = Decoder_RNN(hidden_size, embedding.embedding_matrix, num_layers=num_layers,
                          use_embedding=True, train_embedding=False, dropout_p=0.1)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Training Network.")

    train_network = TrainNetwork(encoder, decoder, index2word, max_length, num_layers=num_layers)
    trainIters(train_network, in_seq, out_seq, lengths, index2word, batch_size=batch_size)

    evaluate_randomly(train_network, in_seq, out_seq, lengths, index2word)
