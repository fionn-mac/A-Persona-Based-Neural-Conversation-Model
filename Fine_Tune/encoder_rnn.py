import torch
import torch.nn as nn
from torch import Tensor

import Pre_Train
from Pre_Train.encoder_rnn import Encoder_RNN as Base_Class

class Encoder_RNN(Base_Class):
    def __init__(self, hidden_size, embedding, num_layers=1, batch_size=1, use_embedding=False, train_embedding=True):
        Base_Class.__init__(self, hidden_size, embedding, num_layers, batch_size, use_embedding, train_embedding)
