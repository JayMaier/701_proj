import io
from torchtext.vocab import build_vocab_from_iterator
import ipdb

file_path = 'en-fr-100.csv'

def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
         for line in f:
             yield line.strip().split()

vocab = build_vocab_from_iterator([yield_tokens(file_path)], specials=['<unk>'])

ipdb.set_trace()