import brain.config as config

import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F

class Word2Vec(object):
	def __init__(self, spo_files):
		self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]


	def create_word_embedding(self):
	        word_set = set()
	        pair_list = []
	        count = 0
	        for spo_path in self.spo_file_paths:
	            print("[Train Word2Vec] Loading spo from {}".format(spo_path))
	            with open(spo_path, 'r', encoding='utf-8') as f:
	                for line in f:
	                    try:
	                        subj, pred, obje = line.strip().split("\t")
	                    except:
	                        print("[Train Word2Vec] Bad spo:", line)
	                    if count % 17 == 0:
	                        word_set.add(subj)
	                        word_set.add(obje)
	                        pair_list.append((subj, obje))
	                    count += 1
	        word_index_dic, inverse_word_dic = self.__get_word_index(word_set)
	        word_size = len(word_set)
	        batch_size = len(pair_list)
	        inputs = [word_index_dic[x[0]] for x in pair_list]
	        labels = [word_index_dic[x[1]] for x in pair_list]

	        embedding_dims = 5
	        W1 = Variable(torch.randn(embedding_dims, word_size).float(), requires_grad=True)
	        W2 = Variable(torch.randn(word_size, embedding_dims).float(), requires_grad=True)
	        num_epochs = 101
	        learning_rate = 0.001

	        for epo in range(num_epochs):
	            loss_val = 0
	            for data, target in zip(inputs, labels):
	                x = Variable(self.get_input_layer(word_size, data)).float()
	                y_true = Variable(torch.from_numpy(np.array([target])).long())

	                z1 = torch.matmul(W1, x)
	                z2 = torch.matmul(W2, z1)

	                log_softmax = F.log_softmax(z2, dim=0)

	                loss = F.nll_loss(log_softmax.view(1,-1), y_true)
	                loss_val += loss.item()
	                loss.backward()
	                W1.data -= learning_rate * W1.grad.data
	                W2.data -= learning_rate * W2.grad.data

	                W1.grad.data.zero_()
	                W2.grad.data.zero_()
	            if epo % 10 == 0:
	                print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
	        return word_index_dic, inverse_word_dic, W1, W2

	def __get_word_index(self, word_set):
	    word_index_dic = dict()
	    inverse_word_dic = dict()
	    for i,word in enumerate(word_set):
	        word_index_dic[word] = i
	        inverse_word_dic[i] = word
	    return word_index_dic, inverse_word_dic

	def get_input_layer(self, word_size, word_idx):
	    x = torch.zeros(word_size).float()
	    x[word_idx] = 1.0
	    return x

	def similarity(self, v,u):
	    return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))