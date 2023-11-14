# https://pytorch.org/tutorials/beginner/torchtext_custom_dataset_tutorial.html

import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.data.utils import get_tokenizer
import spacy
import ipdb
import time
import torch

from torchtext.vocab import build_vocab_from_iterator

# file_path = 'data/deu.txt'
# data_pipe = dp.iter.IterableWrapper([file_path])
# data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
# data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter = '\t', as_tuple=True)


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
            

def dropidx(row):
    return row[1:]

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')



# file_path = 'en-fr-100000.csv'
file_path = 'data/archive/en-fr.csv'
data_pipe = dp.iter.IterableWrapper([file_path])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
# data_pipe = data_pipe.map(dropidx)

# for sample in data_pipe:
#     # ipdb.set_trace()
#     if len(sample) != 2:
#         print(sample)
    
    
    # break

print('building vocab now\n\n'),
start = time.time()

en_vocab = build_vocab_from_iterator(getTokens(data_pipe, 0),
                                     min_freq = 2,
                                     specials = ['<pad>', '<sos>', '<eos>', '<unk>'],
                                     special_first=True)
en_vocab.set_default_index(en_vocab['<unk>'])

torch.save(en_vocab, 'full_en.pth')

print('time to build vocab: ', time.time() - start)
ipdb.set_trace()