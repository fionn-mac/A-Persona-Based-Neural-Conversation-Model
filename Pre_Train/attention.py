import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        hidden : Previous hidden state of the Decoder (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        encoder_outputs: Outputs from Encoder (Sequence Length x Batch Size x Hidden Size)

        return: Attention energies in shape (Batch Size x Sequence Length)
        '''
        max_len = encoder_outputs.size(0) # Encoder Outputs -> L, B, V
        batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs.transpose(0, 1)) # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B, L, 2H]->[B, L, H]
        energy = energy.transpose(2, 1) # [B, H, L]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1) #[B, 1, H]
        energy = torch.bmm(v, energy).squeeze(1) # [B, L]
        return energy
