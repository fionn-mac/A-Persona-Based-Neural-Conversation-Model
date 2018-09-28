import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import Pre_Train
from Pre_Train.decoder_rnn import Decoder_RNN as Base_Class
from attention import Attention

class Decoder_RNN(Base_Class):
    def __init__(self, hidden_size, embedding, personas, num_layers=1, use_embedding=False,
                 train_embedding=True, dropout_p=0.1):
        Base_Class.__init__(self, hidden_size, embedding, num_layers, use_embedding, train_embedding, dropout_p)
        self.personas = nn.Embedding(personas[0], personas[1])
        if use_embedding:
            self.input_size = embedding.shape[1] + personas[1] # Size of embedding vector
        else:
            self.input_size = embedding[1] + personas[1] # Size of embedding vector

        self.gru = nn.GRU(self.hidden_size + self.input_size, self.hidden_size, self.num_layers)

    def forward(self, input, speakers, hidden, encoder_outputs):
        '''
        input           -> (1 x Batch Size)
        speakers        -> (1 x Batch Size, Addressees of inputs to Encoder)
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        encoder_outputs -> (Max Sentence Length, Batch Size, Hidden Size)
        '''
        batch_size = input.size()[1]
        embedded = self.embedding(input) # (1, B, V)
        persona = self.personas(speakers) # (1, B, V')

        features = torch.cat((embedded, persona), 2)

        attn_weights = self.attn(hidden[-1], encoder_outputs)
         # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        rnn_input = torch.cat((features, context), 2)
        output, hidden = self.gru(rnn_input, hidden)

        output = output.squeeze(0) # (1, B, V) -> (B, V)

        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden, attn_weights
