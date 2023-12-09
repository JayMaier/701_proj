import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os
import torch

# Set up a custom iterator to go through partitioned sentences
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

# Create vocab map
vocab = torch.load("en_vocab_500.pth").get_itos()
vocab_map = {}
freq = 0
for w in vocab[::-1]:
    vocab_map[w] = freq
    freq+=1

model = Word2Vec(vector_size=100, window=5, workers=4, negative=2, sg=1)
model.build_vocab_from_freq(vocab_map, keep_raw_vocab=True)

model.train(MySentences("sentencesEn"), total_words=len(vocab_map), epochs=20)
model.save("embeddingsEn.model")