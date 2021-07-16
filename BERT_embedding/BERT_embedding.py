import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

class BertEmbedding():
	def __init__(self,model_dir='chinese_wwm_ext_pytorch'):
		# Load pre-trained model tokenizer (vocabulary)
		self.tokenizer = BertTokenizer.from_pretrained(model_dir)
		self.model = BertModel.from_pretrained(model_dir, output_hidden_states = True,) # Whether the model returns all hidden-states.
		self.model.eval()
		

	def run(self,input_words):
		if len(input_words)!=2: raise ValueError("input must be a list of two strings")
		
		for word in input_words:
			if not isinstance(word,str): raise ValueError("input must be a list of two strings")
		inputs_embedding = []
		for input_word in input_words:
			tokenized_input_word = self.Tokenize(input_word)
			input_word_ids = self.Convert_to_id(tokenized_input_word)
			input_word_seg = self.Get_segement(len(tokenized_input_word))
			hidden_states = self.Get_hidden_states(input_word_ids,input_word_seg)
			token_embeddings = self.Remove_batch_layer(hidden_states) #torch.Size([13, num oftokens, 768])
			final_embedding = self.Get_sentence_embedding(token_embeddings)
			inputs_embedding.append(final_embedding)
		return self.Get_cosine_similarity(inputs_embedding)


	def Get_segement(self,len_):
		return torch.tensor([[0] * len_])

	def Tokenize(self,word):
		marked_text = self.Mark_CLS_SEP(word)
		return self.tokenizer.tokenize(marked_text) #list
	
	# Map the token strings to their vocabulary indeces.
	def Convert_to_id(self,tokenized_text):
		return torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])

	def Mark_CLS_SEP(self,text):
		return "[CLS]" + text + "[SEP]"

	def Get_hidden_states(self,tokens_tensors, segments_tensors):
		with torch.no_grad():
		    outputs = self.model(tokens_tensors, segments_tensors)
		    return outputs[2] #hidden_states

	def Remove_batch_layer(self,hidden_states):
		to_torch = torch.stack(hidden_states, dim=0)
		token_embeddings = torch.squeeze(to_torch, dim=1)
		#token_embeddings = token_embeddings.permute(1,0,2) #change the order
		return token_embeddings

	def Get_sentence_embedding(self,token_embeddings):
		token_embeddings = token_embeddings[2:]
		tmp =[]
		for tokens in token_embeddings:
		    tmp.append(torch.mean(tokens, dim=0))
		layer_to_hidden_state = torch.stack(tmp, dim=0)
		final_embedding = torch.mean(layer_to_hidden_state,dim=0)
		return final_embedding

	def Get_cosine_similarity(self,inputs_embedding):
		word_one, word_two = inputs_embedding
		cosine_similarity = 1 - cosine(word_one,word_two)
		return cosine_similarity

if __name__=="__main__":
	BE = BertEmbedding()
	input1 = '你好啊'
	input2 = '妳好啊'
	ans = BE.run([input1,input2])
	print(ans)


