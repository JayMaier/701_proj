import torchtext
import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import ipdb
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer

from collections import Counter
from torchtext.vocab import Vocab

TEXT = data.Field()
LABEL = data.LabelField(dtype = torch.long)
legacy_train, legacy_test = datasets.IMDB.splits(TEXT, LABEL)

legacy_examples = legacy_train.examples


train_iter, test_iter = IMDB(split=('train', 'test'))


tokenizer = get_tokenizer('basic_english')

counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
    print('line/n')


print(legacy_examples[0].text, legacy_examples[0].label)
ipdb.set_trace()

