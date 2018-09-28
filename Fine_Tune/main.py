from os import path
import sys
sys.path.append('../')

import argparse

import torch

from data_preprocess import Data_Preprocess
from embedding_google import Get_Embedding
from encoder_rnn import Encoder_RNN
from decoder_rnn import Decoder_RNN
from train_network import Train_Network
from run_iterations import Run_Iterations

use_cuda = torch.cuda.is_available()

def load_weights(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state or own_state[name].size() != param.size():
             continue

        # Backwards compatibility for serialized parameters.
        if isinstance(param, torch.nn.Parameter):
            param = param.data

        own_state[name].copy_(param)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_iters", type=int, help="Number of iterations over the training set.", default=7)
    parser.add_argument("-nl", "--num_layers", type=int, help="Number of layers in Encoder and Decoder", default=3)
    parser.add_argument("-z", "--hidden_size", type=int, help="GRU Hidden State Size", default=256)
    parser.add_argument("-pz", "--persona_size", type=int, help="Persona Vector Size", default=128)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate of optimiser.", default=0.001)

    parser.add_argument("-l0", "--min_length", type=int, help="Minimum Sentence Length.", default=5)
    parser.add_argument("-l1", "--max_length", type=int, help="Maximum Sentence Length.", default=20)
    parser.add_argument("-f", "--fold_size", type=int, help="Size of chunks into which training data must be broken.", default=500000)
    parser.add_argument("-tm", "--track_minor", type=bool, help="Track change in loss per cent of Epoch.", default=True)
    parser.add_argument("-tp", "--tracking_pair", type=bool, help="Track change in outputs over a randomly chosen sample.", default=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset directory.", default='../Datasets/Friends/')
    parser.add_argument("-e", "--embedding_file", type=str, help="File containing word embeddings.", default='../../Embeddings/GoogleNews/GoogleNews-vectors-negative300.bin.gz')

    args = parser.parse_args()

    print('Model Parameters:')
    print('Hidden Size                   :', args.hidden_size)
    print('Batch Size                    :', args.batch_size)
    print('Number of Layers              :', args.num_layers)
    print('Max. input length             :', args.max_length)
    print('Learning rate                 :', args.learning_rate)
    print('Number of Epochs              :', args.num_iters)
    print('--------------------------------------------\n')

    print('Reading input data.')
    data_p = Data_Preprocess(args.dataset, min_length=args.min_length, max_length=args.max_length)
    personas = len(data_p.people) + 1

    print("Number of training Samples    :", len(data_p.x_train))
    print("Number of validation Samples  :", len(data_p.x_val))
    print("Number of Personas            :", personas)

    print('Creating Word Embedding.')

    ''' Use pre-trained word embeddings '''

    embedding = Get_Embedding(data_p.word2index, data_p.word2count, args.embedding_file)

    encoder = Encoder_RNN(args.hidden_size, embedding.embedding_matrix, batch_size=args.batch_size,
                          num_layers=args.num_layers, use_embedding=True, train_embedding=False)
    decoder = Decoder_RNN(args.hidden_size, embedding.embedding_matrix, (personas, args.persona_size),
                          num_layers=args.num_layers, use_embedding=True, train_embedding=False, dropout_p=0.1)

    # Delete embedding object post weight initialization in encoder and decoder
    del embedding

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    if path.isfile('../Pre-Train/encoder.pt') and path.isfile('../Pre-Train/decoder.pt'):
        load_weights(encoder, torch.load('../Pre-Train/encoder.pt'))
        load_weights(decoder, torch.load('../Pre-Train/decoder.pt'))

    print("Training Network.")

    train_network = Train_Network(encoder, decoder, data_p.index2word, num_layers=args.num_layers)

    run_iterations = Run_Iterations(train_network, data_p.x_train, data_p.y_train, data_p.train_lengths, data_p.train_speakers,
                                    data_p.train_addressees, data_p.index2word, args.batch_size, args.num_iters, args.learning_rate,
                                    fold_size=args.fold_size, track_minor=args.track_minor, tracking_pair=args.tracking_pair,
                                    val_in_seq=data_p.x_val, val_out_seq=data_p.y_val, val_lengths=data_p.val_lengths,
                                    val_speakers=data_p.val_speakers, val_addressees=data_p.val_addressees)
    run_iterations.train_iters()
    run_iterations.evaluate_randomly()
