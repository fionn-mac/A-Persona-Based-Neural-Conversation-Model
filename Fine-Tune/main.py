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
    parser.add_argument("-z", "--hidden_size", type=int, help="GRU Hidden State Size", default=1024)
    parser.add_argument("-p", "--persona_size", type=int, help="Persona Embedding Size", default=100)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate of optimiser.", default=0.001)

    parser.add_argument("-l", "--max_length", type=int, help="Maximum Sentence Length.", default=20)
    parser.add_argument("-tp", "--tracking_pair", type=bool, help="Track change in outputs over a randomly chosen sample.", default=False)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset directory.", default='../Datasets/Friends/')
    parser.add_argument("-e", "--embedding_file", type=str, help="File containing word embeddings.", default='../../Embeddings/GoogleNews/GoogleNews-vectors-negative300.bin.gz')

    args = parser.parse_args()

    num_iters = args.num_iters
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    persona_size = args.persona_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_length = args.max_length
    tracking_pair = args.tracking_pair
    dataset = args.dataset
    embedding_file = args.embedding_file

    print('Model Parameters:')
    print('Hidden Size                  :', hidden_size)
    print('Persona Size                 :', persona_size)
    print('Batch Size                   :', batch_size)
    print('Number of Layers             :', num_layers)
    print('Max. input length            :', max_length)
    print('Learning rate                :', learning_rate)
    print('--------------------------------------\n')

    print('Reading input data.')
    data_preprocess = Data_Preprocess(dataset, max_length=max_length)

    train_in_seq = data_preprocess.x_train
    train_out_seq = data_preprocess.y_train
    train_lengths = data_preprocess.lengths_train
    train_speakers = data_preprocess.speaker_list_train
    train_addressees = data_preprocess.addressee_list_train

    dev_in_seq = data_preprocess.x_val
    dev_out_seq = data_preprocess.y_val
    dev_lengths = data_preprocess.lengths_val
    dev_speakers = data_preprocess.speaker_list_val
    dev_addressees = data_preprocess.addressee_list_val

    word2index = data_preprocess.word2index
    index2word = data_preprocess.index2word
    word2count = data_preprocess.word2count
    vocab_size = data_preprocess.vocab_size
    personas = len(data_preprocess.people) + 1

    print("Number of training Samples    :", len(train_in_seq))
    print("Number of validation Samples  :", len(dev_in_seq))
    print("Number of Personas            :", personas)

    print('Creating Word Embedding.')

    ''' Use pre-trained word embeddings '''
    embedding = Get_Embedding(word2index, word2count, embedding_file)

    encoder = Encoder_RNN(hidden_size, embedding.embedding_matrix, batch_size=batch_size,
                          num_layers=num_layers, use_embedding=True, train_embedding=False)
    decoder = Decoder_RNN(hidden_size, embedding.embedding_matrix, (personas, persona_size), num_layers=num_layers,
                          use_embedding=True, train_embedding=False, dropout_p=0.1)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    load_weights(encoder, torch.load('../Pre-Train/encoder.pt'))
    load_weights(decoder, torch.load('../Pre-Train/decoder.pt'))

    print("Training Network.")

    train_network = Train_Network(encoder, decoder, index2word, num_layers=num_layers)

    run_iterations = Run_Iterations(train_network, train_in_seq, train_out_seq, train_lengths, train_speakers,
                                    train_addressees, index2word, batch_size, num_iters, learning_rate,
                                    tracking_pair=tracking_pair, dev_in_seq=dev_in_seq, dev_out_seq=dev_out_seq,
                                    dev_input_lengths=dev_lengths, dev_speakers=dev_speakers, dev_addressees=dev_addressees)

    run_iterations.train_iters()
    run_iterations.evaluate_randomly()
