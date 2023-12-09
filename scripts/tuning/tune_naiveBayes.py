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

# Hyper-params for tuning
ncontext = 2 # Fixed at all times
vocab_size = 10000
use_bigrams = False

# Load vocabularies
vocabEn = pd.read_csv("vocabs/unigrams_en_top20k.csv", index_col=0)[:vocab_size]

vocabFr = pd.read_csv("vocabs/unigrams_fr_top20k.csv", index_col=0)[:vocab_size]
if use_bigrams:
    bigrams = pd.read_csv("vocabs/bigrams_fr_top20k.csv", index_col=0)[:vocab_size]
    vocabFr = pd.concat([vocabFr, bigrams])

def in_vocabFr(x):
    try:
        vocabFr.loc[x]
        return x
    except:
        return '<UNK>'
    
def in_vocabEn(x):
    try:
        vocabEn.loc[x]
        return x
    except:
        return '<UNK>'
    

train_size = int(0.9 * 22.5 * 10 ** 6)
tuning_train_size = 10 ** 6
tuning_test_size = 10 ** 5

fr_spacy = spacy.load('fr_core_news_sm')
en_spacy = spacy.load('en_core_web_sm')

def frTokenize(text):
    unigrams = [str.lower(token.text) for token in fr_spacy.tokenizer(text) if re.fullmatch(r"[a-zàâçéèêîôûù'`]+", str.lower(token.text)) is not None]
    # filter by vocab to get only the valus used for prediction
    words = list(map(lambda x: in_vocabFr(x), unigrams))
    if len(words) < ncontext:
        return []
    adj_words = [','.join(pair) for pair in sliding_window_view(words, ncontext)]
    dist1_words = []
    if len(unigrams) > ncontext:
        dist1_words = [pair[0] + "," + pair[2] for pair in sliding_window_view(words, ncontext + 1)]
    if len(unigrams) <= ncontext or not use_bigrams:
        return adj_words + dist1_words
    bigrams = [' '.join(pair) for pair in sliding_window_view(unigrams[1:], ncontext)]
    # filter bigrams by vocab
    bigrams = list(map(lambda x: in_vocabFr(x), bigrams))
    adj_bigrams = [','.join([words[i], bigrams[i]]) for i in range(len(bigrams))]
    dist1_bigrams = [','.join([words[i], bigrams[i+1]]) for i in range(len(bigrams) - 1)]
    return adj_words + dist1_words + adj_bigrams + dist1_bigrams

def tokenize(textEn, textFr):
    y = [str.lower(token.text) for token in fr_spacy.tokenizer(textFr) if re.fullmatch(r"[a-zàâçéèêîôûù'`]+", str.lower(token.text)) is not None]
    unigrams = [str.lower(token.text) for token in en_spacy.tokenizer(textEn) if re.fullmatch(r"[a-z'`]+", str.lower(token.text)) is not None]
    # filter by vocab to get only the values used for prediction
    y = list(map(lambda x: in_vocabFr(x), y))
    unigrams = list(map(lambda x: in_vocabEn(x), unigrams))
    
    eos = min(len(y), len(unigrams))
    tokens = [','.join([unigrams[i], y[i]]) for i in range(eos)]
    tokens += [','.join([unigrams[i-1], y[i]]) for i in range(1, eos)]
    tokens += [','.join([unigrams[i-2], y[i]]) for i in range(2, eos)]

    if len(y) <= ncontext or not use_bigrams:
        return tokens
    
    bigrams = [' '.join(pair) for pair in sliding_window_view(y, ncontext)]
    bigrams = list(map(lambda x: in_vocabFr(x), bigrams))
    tokens += [','.join([unigrams[i], bigrams[i]]) for i in range(eos - 1)]
    tokens += [','.join([unigrams[i], bigrams[i+1]]) for i in range(eos - 2)]
    tokens += [','.join([unigrams[i], bigrams[i+2]]) for i in range(eos - 3)]
    return tokens

def getTokens(data_iter):
    for english, french in data_iter:
        # yield (frTokenize(french), None)
        yield (None, tokenize(english, french))
        # yield (frTokenize(french), tokenize(english, french))
            
if __name__ == '__main__':
    start = time.time()
    file_path = 'tuning_train_data.csv'
    # file_path = '../archive/en-fr_100k.csv'
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    train_pipe = data_pipe.parse_csv(skip_lines=600001, delimiter=',', as_tuple=True)
    
    pipetime = time.time()
    counterXfr = Counter()
    counterXen = Counter()
    
    i = 1
    for tokensXfr, tokensXen in tqdm(getTokens(train_pipe)):
        counterXfr.update(tokensXfr)
        counterXen.update(tokensXen)
        if i % (10 ** 5) == 0:
            print("Saving checkpoint in", i)
            torch.save(counterXfr, "tuning/counters/counter_Xfr_given_Y_" + str(i) + ".pth")
            torch.save(counterXen, "tuning/counters/counter_Xen_given_Y_" + str(i) + ".pth")
        i += 1
    torch.save(counterXfr, "counter_Xfr_given_Y.pth")
    torch.save(counterXen, "counter_Xen_given_Y_.pth")
    entime = time.time()
    print('\n Time to count frequencies: ', entime - pipetime)

    counterXfr = torch.load("counter_Xen_given_Y.pth")
    freq = pd.DataFrame(counterXfr.items(), columns=['y', 'freq'])
    freq = freq[~freq['y'].str.contains('<UNK>')]
    freq['y'] = freq['y'].str.split(",", n=1)
    freq['Xfr'] = freq['y'].apply(lambda x: x[0])
    freq['Y'] = freq['y'].apply(lambda x: x[1])
    del freq['y']
    pivot = pd.pivot_table(freq, values='freq', index='Y', columns=['Xfr'], aggfunc='sum', fill_value=1)
    pivot.to_csv("tuning/Xfr_given_Y.csv")
    
    counterXfr = torch.load("counter_Xen_given_Y.pth")
    freq = pd.DataFrame(counterXen.items(), columns=['y', 'freq'])
    freq = freq[~freq['y'].str.contains('<UNK>')]
    freq['y'] = freq['y'].str.split(",", n=1)
    freq['Xfr'] = freq['y'].apply(lambda x: x[0])
    freq['Y'] = freq['y'].apply(lambda x: x[1])
    del freq['y']
    pivot = pd.pivot_table(freq, values='freq', index='Y', columns=['Xfr'], aggfunc='sum', fill_value=1)
    pivot.to_csv("tuning/Xen_given_Y.csv")

# Note: After tuning script has generated all different model variants,
# naiveBayes.py is used to run inference on each setting separately and report BLEU scores.