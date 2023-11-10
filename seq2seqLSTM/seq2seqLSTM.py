'''
This program translates english to french using a seq2seq approach where
the encoder is a vanilla LSTM and the decoder is a vanilla LSTM.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator, TabularDataset
import torchtext
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split

### Preprocessing for seq2seq

nrows = 1000
df = pd.read_csv("../../data/en-fr.csv", nrows=nrows)

train, test = train_test_split(df, test_size=0.1)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

french = Field(sequential=True, tokenize=tokenizer_ger, use_vocab=True, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(sequential=True, tokenize=tokenizer_en, use_vocab=True, lower=True, init_token='<sos>', eos_token='<eos>')

fields = {'en': ('English', english), 'fr': ('French', french)}

train_data, test_data = TabularDataset.splits(
    path='',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=fields)

english.build_vocab(train_data, max_size=10000, min_freq=2)
french.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    device='mps')

for batch in train_iterator:
    print(batch)



