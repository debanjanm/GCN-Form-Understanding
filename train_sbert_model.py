##===========================================================================================================##
from sentence_transformers import SentenceTransformer

model_path  = "C://Users//acer//OneDrive//Documents//GitHub//GCN-Form-Understanding//model//sbert-local//"
model = SentenceTransformer(model_path)

# sentences = 'This framework generates embeddings for each input sentence'
# embeddings = model.encode(sentences)

##===========================================================================================================##
text_file_path = "text_data.txt"

with open(text_file_path, "r") as f:
    text = f.read()

# tokenizer = model.tokenizer
# word_embedding_model = model._first_module()

# new_tokens = text.split()

# new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

# tokenizer.add_tokens(list(new_tokens))

# model.tokenizer = tokenizer
# word_embedding_model.resize_token_embeddings(len(tokenizer))

tokens = text.split()
word_embedding_model = model._first_module()   #Your models.Transformer object
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))


from sentence_transformers import models
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

sentences = 'This framework generates embeddings for each input sentence'
embeddings = model.encode(sentences)

##===========================================================================================================##
import pickle

pickle.dump(model, open("sentence_transformer_v1.p", "wb"))

s = pickle.load(open("sentence_transformer_v1.p", "rb"))
s.encode(sentences)
##===========================================================================================================##
