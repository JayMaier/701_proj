import pandas as pd
import torchdata.datapipes as dp
from torchtext.vocab import vocab

import spacy
import re
import time
import torch
from tqdm import tqdm
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view

# Use an iterator to parse the dataset
# For each parsed sentence, tokenize the english and french sentence and add each word pair - according to the context window - to a counter
# Use only values with frequency at least 500 and add special tokens to create the vocab

min_freq = 500
ngrams = 2
trainSize = int(0.9 * 22.5 * 10 ** 6)

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

def enTokenize(text):
    unigrams = [str.lower(token.text) for token in en_spacy.tokenizer(text) if re.fullmatch(r"[a-z]+", str.lower(token.text)) is not None]
    if len(unigrams) >= ngrams:
        return [' '.join(pair) for pair in sliding_window_view(unigrams, ngrams)]
    return ['']


def frTokenize(text):
    unigrams = [str.lower(token.text) for token in fr_spacy.tokenizer(text) if re.fullmatch(r"[a-zàâçéèêîôûù]+", str.lower(token.text)) is not None]
    if len(unigrams) >= ngrams:
        return [' '.join(pair) for pair in sliding_window_view(unigrams, ngrams)]
    return ['']


def getTokens(data_iter, place):
    for english, french in data_iter:
        if place == 0:
            yield enTokenize(english)
        else:
            yield frTokenize(french)
            
if __name__ == '__main__':
    start = time.time()
    file_path = 'archive/en-fr.csv'
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
    data_pipe = data_pipe.header(trainSize)
    
    pipetime = time.time()
    print('\n Time to set up pipeline: ', pipetime - start)
    print('Dataset loaded, building English vocab now...')
    
    en_counter = Counter()
    fr_counter = Counter()
    
    
    for tokens in tqdm(getTokens(data_pipe, 0)):
        en_counter.update(tokens)
    en_vocab = vocab(en_counter, min_freq=min_freq, specials=('<UNK>', '<BOS>', '<EOS>', '<PAD>'))
    unigrams = pd.Series(en_counter.values(), index=en_counter.keys())
    del unigrams['']
    unigrams = unigrams.sort_values()[::-1][:20000]
    unigrams.to_csv("unigrams_en_top20k.csv")
    torch.save(en_vocab, 'unigrams_en_top20k.csv')
    torch.save(en_counter, "counter_en_bigram.pth")
    entime = time.time()
    print('\n Time to get english vocab: ', entime - pipetime)
    
    print('\n English vocab saved, starting french now')
    
    for tokens in tqdm(getTokens(data_pipe, 1)):
        fr_counter.update(tokens)
    fr_vocab = vocab(fr_counter, min_freq=min_freq, specials=('<UNK>', '<BOS>', '<EOS>', '<PAD>'))
    bigrams = pd.Series(fr_counter.values(), index=fr_counter.keys())
    del bigrams['']
    bigrams = bigrams.sort_values()[::-1][:20000]
    bigrams.to_csv("bigrams_fr_top20k.csv")
    torch.save(fr_vocab, 'fr_vocab_500_bigram.pth')
    torch.save(fr_counter, "counter_fr_bigram.pth")