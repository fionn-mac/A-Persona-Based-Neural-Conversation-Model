import torch
import numpy as np
from gensim.models import KeyedVectors

use_cuda = torch.cuda.is_available()

class Get_Embedding(object):
    def __init__(self, word_index, word_count, file_path, vocab_size=None):
        self.file_path = file_path
        self.embedding_matrix = self.create_embed_matrix(word_index, word_count,
                                                         vocab_size)

    def create_embed_matrix(self, word_index, word_count, vocab_size):
        ''' Assuming embedding to be in the form of KeyedVectors '''
        word2vec = KeyedVectors.load_word2vec_format(self.file_path, binary=True)

        # Fix embedding dimensions.
        num_words = len(word_index) + 1
        if vocab_size: num_words = min(vocab_size, num_words)

        embedding_matrix = np.zeros((num_words, 300))

        for word, i in word_index.items():
            # Words not found in embedding index will be all-zeros.
            if word not in word2vec.vocab:
                continue
            embedding_matrix[i] = word2vec.word_vec(word)

        del word2vec

        embedding_matrix = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
        if use_cuda: embedding_matrix = embedding_matrix.cuda()

        return embedding_matrix
