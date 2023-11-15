import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os
import torch

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

# Create vocab map
# vocab = pd.read_csv("vocabEn.csv")
# vocab = vocab[:20000] # keep the 20k most frequent words
# vocab_map = vocab.set_index('en').to_dict()['count']
vocab = torch.load("en_vocab.pth").get_itos()


model = Word2Vec(vector_size=100, window=5, workers=4, negative=2, sg=1)
model.build_vocab(vocab)
# model = Word2Vec.load("word2vec_0.model")

model.train(MySentences("sentencesEn"), total_words=len(vocab), epochs=20)
model.save("embeddingsEn.model")