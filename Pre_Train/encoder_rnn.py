import torch
import torch.nn as nn
from torch import Tensor

class Encoder_RNN(nn.Module):
    def __init__(self, hidden_size, embedding, num_layers=1, batch_size=1, use_embedding=False, train_embedding=True):
        super(Encoder_RNN, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # V - Size of embedding vector

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1]

        self.embedding.weight.requires_grad = train_embedding

        self.gru = nn.GRU(self.input_size, hidden_size, self.num_layers, bidirectional=True)

    def forward(self, input, input_lengths, hidden):
        '''
        input           -> (Max. Sequence Length (per batch) x Batch Size)
        input_lengths   -> (Batch Size (Sorted in decreasing order of lengths))
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        '''
        embedded = self.embedding(input) # L, B, V

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Concatenate Bidirectional Outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

    def init_hidden(self, batch_size=0):
        if batch_size == 0: batch_size = self.batch_size
        result = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            return result.cuda()
        else:
            return result
