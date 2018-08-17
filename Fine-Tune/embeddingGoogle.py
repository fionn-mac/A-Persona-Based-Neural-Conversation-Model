import torch
import numpy as np
from gensim.models import KeyedVectors

use_cuda = torch.cuda.is_available()

class GetEmbedding(object):
    def __init__(self, word_index, word_count, dir_path, vocab_size=100000):
        self.dir_path = dir_path
        self.embedding_matrix = self.create_embed_matrix(word_index, word_count,
                                                         vocab_size)

    def create_embed_matrix(self, word_index, word_count, vocab_size):
        print('Preparing Embedding Matrix.')

        file_name = self.dir_path + 'GoogleNews-vectors-negative300.bin.gz'
        word2vec = KeyedVectors.load_word2vec_format(file_name, binary=True)
        # prepare embedding matrix
        num_words = min(vocab_size, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, 300))

        for word, i in word_index.items():
            # words not found in embedding index will be all-zeros.
            if i >= vocab_size or word not in word2vec.vocab:
                continue
            embedding_matrix[i] = word2vec.word_vec(word)

        del word2vec

        embedding_matrix = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
        if use_cuda: embedding_matrix = embedding_matrix.cuda()

        return embedding_matrix
