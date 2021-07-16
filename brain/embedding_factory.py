from brain.BERT_embedding import BertEmbedding
from brain.word_embedding_skip_gram import Word2Vec
from brain.word_vectors_Chinese_Word_Vectors import ChineseWordVector

import os

class EmbeddingFactory():
    def __init__(self,embedding_type,folder_name):
        self.embedding = self.init(embedding_type,folder_name)

    def init(self,embedding_type, folder_name):
        path = self.get_model_path(str(folder_name))
        if embedding_type == "Bert":
            return BertEmbedding(model_name=path)
        elif embedding_type == "ChineseWordVector":
            return ChineseWordVector('word_vectors/merge_sgns_bigram_char300.txt', 0)
        # else:
        #     self.embedding = Word2Vec(spo_files)

    def get_model_path(self,model_name):
        FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(FILE_DIR_PATH,'..',"models",model_name)