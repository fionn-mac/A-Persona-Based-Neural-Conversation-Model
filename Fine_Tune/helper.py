import time
import math

import torch
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import Pre_Train
from Pre_Train.helper import Helper as Base_Class

class Helper(Base_Class):
    def __init__(self):
        Base_Class.__init__(self)
