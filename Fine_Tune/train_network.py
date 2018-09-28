import random
import torch

import Pre_Train
from Pre_Train.train_network import Train_Network as Base_Class

class Train_Network(Base_Class):
    def __init__(self, encoder, decoder, index2word, num_layers=1, teacher_forcing_ratio=0.5):
        Base_Class.__init__(self, encoder, decoder, index2word, num_layers, teacher_forcing_ratio)
