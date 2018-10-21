import torch
import numpy as np
from gensim.models import KeyedVectors

use_cuda = torch.cuda.is_available()

import Pre_Train
from Pre_Train.embedding_google import Get_Embedding as Base_Class

class Get_Embedding(Base_Class):
    def __init__(self, word_index, file_path):
        Base_Class.__init__(self, word_index, file_path)
