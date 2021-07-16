import numpy as np
import argparse
import random

class ChineseWordVector(object):
    def __init__(self, vector_file, topn = 10000):
        self.vector_file = vector_file
        self.topn = topn
        self.vector_matrix, self.index2word, self.word2index = self.get_vector_matrix()

    def get_vector_matrix():
        vectors, iw, wi, dim = self.read_vectors()
        # Turn vectors into numpy format and normalize them
        matrix = np.zeros(shape=(len(iw), dim), dtype=np.float32)
        for i, word in enumerate(iw):
            matrix[i, :] = vectors[word]
        matrix = normalize(matrix)
        return matrix, iw, wi

    def read_vectors():  # read top n word vectors, i.e. top is 10000
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
        with open(self.vector_file, encoding='utf-8', errors='ignore') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    dim = int(line.rstrip().split()[1])
                    continue
                lines_num += 1
                tokens = line.rstrip().split(' ')
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0])
                if self.topn != 0 and lines_num >= self.topn:
                    break
        for i, w in enumerate(iw):
            wi[w] = i
        return vectors, iw, wi, dim

    def normalize(matrix):
        norm = np.sqrt(np.sum(matrix * matrix, axis=1))
        matrix = matrix / norm[:, np.newaxis]
        return matrix

    def get_similarity_matrix(word):
        if word in self.word2index:
            matrix = self.vector_matrix[word2index[word]]