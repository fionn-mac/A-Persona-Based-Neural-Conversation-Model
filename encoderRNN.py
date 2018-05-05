import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, batch_size=1, use_embedding=False,
                 train_embedding=True):
        super(EncoderRNN, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # V - Size of embedding vector

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1]

        self.embedding.weight.requires_grad = train_embedding

        self.gru = nn.GRU(self.input_size, hidden_size, bidirectional=True)

    def forward(self, input, input_lengths, hidden):
        '''
        input           -> (Max. Sequence Length (per batch) x Batch Size)
        input_lengths   -> (Batch Size (Sorted in decreasing order of lengths))
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        '''
        embedded = self.embedding(input) # L, B, V

        # packed:
        #       - data: (sum(batch_sizes), word_vec_size)
        #       - batch_sizes: list of batch sizes
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        outputs, hidden = self.gru(packed, hidden)
        # output_lens == input_lengths
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Concatenate Bidirectional Outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

    def initHidden(self, batch_size=0):
        if batch_size == 0: batch_size = self.batch_size
        result = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result
