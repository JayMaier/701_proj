import torchdata.datapipes as dp
from torchtext.vocab import vocab

import spacy
import ipdb
import time
import torch
from tqdm import tqdm
from collections import Counter



en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

def enTokenize(text):
    return [token.text for token in en_spacy.tokenizer(text)]

def frTokenize(text):
    return [token.text for token in fr_spacy.tokenizer(text)]

def getTokens(data_iter, place):
    for english, french in data_iter:
        if place == 0:
            yield enTokenize(english)
        else:
            yield frTokenize(french)
            
if __name__ == '__main__':
    start = time.time()
    # file_path = 'data/archive/en-fr.csv'
    file_path = 'en-fr-1000000.csv'
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
    
    
    pipetime = time.time()
    # print('\n Time to set up pipeline: ', pipetime - start)
    print('Dataset loaded, building English vocab now...')
    
    en_counter = Counter()
    fr_counter = Counter()
    
    
    for tokens in tqdm(getTokens(data_pipe, 0)):
        en_counter.update(tokens)
    # ipdb.set_trace()
    en_vocab = vocab(en_counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
    torch.save(en_vocab, '../models/en_vocab.pth')
    # ipdb.set_trace()
    entime = time.time()
    print('\n Time to get english vocab: ', entime - pipetime)
    
    print('\n English vocab saved, starting french now')
    
    for tokens in tqdm(getTokens(data_pipe, 1)):
        fr_counter.update(tokens)
    # ipdb.set_trace()
    fr_vocab = vocab(fr_counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
    torch.save(fr_vocab, '../models/fr_vocab.pth')
    
    print('\n French vocab saved, time for french vocab: ', time.time() - entime)