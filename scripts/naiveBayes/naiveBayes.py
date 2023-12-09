import pandas as pd
import numpy as np
import torchdata.datapipes as dp
from torchtext.vocab import vocab

import spacy
import re
import time
import torch
from tqdm import tqdm
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view
from torchtext.data.metrics import bleu_score

# Hyper-parameters
ncontext = 2 # Always fixed
vocab_size = 10000
use_bigrams = True

# Load tokenizers
en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

# Load vocabularies
vocabEn = pd.read_csv("vocabs/unigrams_en_top20k.csv", index_col=0)[:vocab_size]

vocabFr = pd.read_csv("vocabs/unigrams_fr_top20k.csv", index_col=0)[:vocab_size]
if use_bigrams:
    bigrams = pd.read_csv("vocabs/bigrams_fr_top20k.csv", index_col=0)[:vocab_size]
    vocabFr = pd.concat([vocabFr, bigrams])

# Use helper functions to filter out tokens that are not part of the vocab
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
    
# Create X_FR | Y frequency table
trainSize = int(0.9 * 22.5 * 10 ** 6)

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
    if len(unigrams) <= ncontext:
        return adj_words + dist1_words
    bigrams = [' '.join(pair) for pair in sliding_window_view(unigrams[1:], ncontext)]
    # filter bigrams by vocab
    bigrams = list(map(lambda x: in_vocabFr(x), bigrams))
    adj_bigrams = [','.join([words[i], bigrams[i]]) for i in range(len(bigrams))]
    dist1_bigrams = [','.join([words[i], bigrams[i+1]]) for i in range(len(bigrams) - 1)]
    return adj_words + dist1_words + adj_bigrams + dist1_bigrams


def getTokens(data_iter):
    for _, french in data_iter:
        yield frTokenize(french)
            
start = time.time()
file_path = '../archive/en-fr.csv'
data_pipe = dp.iter.IterableWrapper([file_path])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
data_pipe = data_pipe.header(trainSize) # Count frequencies only for training dataset

pipetime = time.time()
counter = torch.load("counters/counter_Xfr_given_Y_bigr.pth")
counter = Counter()

i = 1
for tokens in tqdm(getTokens(data_pipe)):
    counter.update(tokens)
    if i % (5 * 10 ** 5) == 0:
        print("Saving checkpoint in", i)
        torch.save(counter, "counters/counter_Xfr_given_Y" + str(i) + ".pth")
    i += 1
torch.save(counter, "counter_Xfr_given_Y.pth")
entime = time.time()
print('\n Time to count frequencies: ', entime - pipetime)

counter = torch.load("counter_Xfr_given_Y.pth")
freq = pd.DataFrame(counter.items(), columns=['y', 'freq'])
freq = freq[~freq['y'].str.contains('<UNK>')]
freq['y'] = freq['y'].str.split(",", n=1)
freq['Xfr'] = freq['y'].apply(lambda x: x[0])
freq['Y'] = freq['y'].apply(lambda x: x[1])
del freq['y']
pivot = pd.pivot_table(freq, values='freq', index='Y', columns=['Xfr'], aggfunc='sum', fill_value=1)
pivot.to_csv("Xfr_given_Y.csv")


# Create X_EN | Y frequency table
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

    if len(y) <= ncontext:
        return tokens
    
    bigrams = [' '.join(pair) for pair in sliding_window_view(y, ncontext)]
    bigrams = list(map(lambda x: in_vocabFr(x), bigrams))
    tokens += [','.join([unigrams[i], bigrams[i]]) for i in range(eos - 1)]
    tokens += [','.join([unigrams[i], bigrams[i+1]]) for i in range(eos - 2)]
    tokens += [','.join([unigrams[i], bigrams[i+2]]) for i in range(eos - 3)]
    return tokens

def getTokens(data_iter):
    for english, french in data_iter:
        yield tokenize(english, french)
            
start = time.time()
file_path = '../archive/en-fr.csv'
data_pipe = dp.iter.IterableWrapper([file_path])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
data_pipe = data_pipe.header(trainSize) # Count frequencies only for training dataset

pipetime = time.time()
counter = Counter()

i = 1
for tokens in tqdm(getTokens(data_pipe)):
    counter.update(tokens)
    if i % (5 * 10 ** 5) == 0:
        print("Saving checkpoint in", i)
        torch.save(counter, "counters/counter_Xen_given_Y" + str(i) + ".pth")
    i += 1
torch.save(counter, "counter_Xen_given_Y.pth")
entime = time.time()
print('\n Time to count frequencies: ', entime - pipetime)

freq = pd.DataFrame(counter.items(), columns=['y', 'freq'])
freq[['Xen', 'Y']] = freq['y'].str.split(",", n=1, expand=True)
del freq['y']
freq = freq[(freq['Xen'] != '<UNK>') & (freq['Y'] != '<UNK>')]
pivot = pd.pivot_table(freq, values='freq', index='Y', columns=['Xen'], aggfunc='sum', fill_value=1)
pivot.to_csv("Xen_given_Y.csv")


### Run inference on trained model
X = pd.read_csv("Xen_given_Y.csv", index_col='Y')
Y = pd.read_csv("Xfr_given_Y.csv", index_col='Y')
X = X.loc[X.index.dropna()]
Y = Y.loc[Y.index.dropna()]

Xen_given_Y = X.divide(X.sum(axis=1), axis=0)
P_Y = Y.sum(axis=1) / Y.sum().sum()
Xfr_given_Y = Y.divide(Y.sum(axis=1), axis=0)

def predictWord(adjEn, adjFr):
    # If word to translate is not in english vocab,
    # return it (might be a number or punctuation mark)
    if in_vocabEn(adjEn[-1]) == '<UNK>':
        return adjEn[-1]
    prob = np.log(P_Y * 10 ** 6) # showed better performance
    for w in adjEn:
        try:
            prob = prob.mul(Xen_given_Y[w], fill_value=0)
        except:
            continue
    for w in adjFr:
        try:
            prob = prob.mul(Xfr_given_Y[w], fill_value=0)
        except:
            continue
    return prob.idxmax()

def translate(sentence):
    pred = []
    for i in range(len(sentence)):
        if i > 2:
            k = 2
        if i == 1:
            k = 1
        else:
            k = 0
        adjEn = sentence[i-k:i+1]
        adjFr = pred[i-k:i]
        pred += [predictWord(adjEn, adjFr)]
    return pred


### Evaluation
# To evaluate the trained model, iterate through the test dataset
# (last 10% cells), transform them into a form that can be predicted
# by naive Bayes - all lowercase, no punctuation marks, etc - and run
# the BLEU score evaluation on each sentence
trainSize = int(0.9 * 22.5 * 10 ** 6)
testSize = 2270377

def getScore(data_iter):
    for english, french in data_iter:
        pred = [translate([str.lower(word) for word in str.split(english)])]
        fr = [[str.lower(word) for word in str.split(french)]]
        s = bleu_score(pred, [fr])
        if s > 0.8:
            print([str.lower(word) for word in str.split(english)])
            print(pred[0])
            print(fr[0])
            print("####################")
        yield s
            
start = time.time()
file_path = '../archive/en-fr.csv'
data_pipe = dp.iter.IterableWrapper([file_path])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
# Run only on test dataset by ignoring the first trainSize lines
data_pipe = data_pipe.parse_csv(skip_lines=trainSize + 1, delimiter = ',', as_tuple=True)
score = []
i = 1
for s in tqdm(getScore(data_pipe)):
    score.append(s)
    i += 1
print("Average score", np.average(score), "with standard deviation", np.std(score))