import torch
import numpy as np
from gensim.models import KeyedVectors

class Get_Embedding(object):
    def __init__(self, word_index, file_path):
        self.file_path = file_path
        self.embedding_matrix = self.create_embed_matrix(word_index)

    def create_embed_matrix(self, word_index):
        ''' Assuming embedding to be in the form of KeyedVectors '''
        word2vec = KeyedVectors.load_word2vec_format(self.file_path, binary=True)

        # Fix embedding dimensions.
        embedding_matrix = np.zeros((len(word_index), 300))

        count = 0
        for word, i in word_index.items():
            # Words not found in embedding index will be all-zeros.
            if word not in word2vec.vocab:
                count += 1
                continue
            embedding_matrix[i] = word2vec.word_vec(word)

        print('%d word were not present in Google Word2Vec' % count)
        del word2vec

        embedding_matrix = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)

        return embedding_matrix
