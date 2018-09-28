import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import Pre_Train
from Pre_Train.attention import Attention as Base_Class

class Attention(Base_Class):
    def __init__(self, method, hidden_size):
        Base_Class.__init__(self, method, hidden_size)
