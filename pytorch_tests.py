## fropm https://github.com/UtkarshGarg-UG/Deep-Learning-Projects/blob/main/NLP/Custom%20Dataset/loading%20custom%20dataset%20(text).ipynb

#sys libs
import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

#data manupulation libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from pandarallel import pandarallel
# Initialization
# pandarallel.initialize()


#string manupulation libs
import re
import string
from string import digits
import spacy

#torch libs
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim


import ipdb
import dask.dataframe as dd
from dask.distributed import Client




class Vocabulary:
    
    '''
    __init__ method is called by default as soon as an object of this class is initiated
    we use this method to initiate our vocab dictionaries
    '''
    def __init__(self, freq_threshold, max_size):
        '''
        freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
        max_size : max source vocab size
        '''
        #initiate the index to token dict
        self.itos = {0: '<PAD>', 1:'<SOS>', 2:'<EOS>', 3: '<UNK>'}
        #initiate the token to index dict
        self.stoi = {k:j for j,k in self.itos.items()} 
        
        self.freq_threshold = freq_threshold
        self.max_size = max_size
    
    '''
    __len__ is used by dataloader later to create batches
    '''
    def __len__(self):
        return len(self.itos)
    
    '''
    a simple tokenizer to split on space and converts the sentence to list of words
    '''
    @staticmethod
    def tokenizer(text):
        return [tok.lower().strip() for tok in text.split(' ')]
    
    '''
    build the vocab: create a dictionary mapping of index to string (itos) and string to index (stoi)
    output ex. for stoi -> {'the':5, 'a':6, 'an':7}
    '''
    def build_vocabulary(self, sentence_list):
        #calculate the frequencies of each word first to remove the words with freq < freq_threshold
        frequencies = {}  #init the freq dict
        idx = 4 #index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk
        
        #calculate freq of words
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                    
                    
        #limit vocab by removing low freq words
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold} 
        
        #limit vocab to the max_size specified
        if len(frequencies)>self.max_size-idx: 
            frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx]) # idx =4 for pad, start, end , unk
            
        #create vocab
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
            
    '''
    convert the list of words to a list of corresponding indexes
    '''    
    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else: #out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi['<UNK>'])
                
        return numericalized_text

class Train_Dataset(Dataset):
    '''
    Initiating Variables
    df: the training dataframe
    source_column : the name of source text column in the dataframe
    target_columns : the name of target text column in the dataframe
    transform : If we want to add any augmentation
    freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
    source_vocab_max_size : max source vocab size
    target_vocab_max_size : max target vocab size
    '''
    
    def __init__(self, df, source_column, target_column, transform=None, freq_threshold = 5,
                source_vocab_max_size = 10000, target_vocab_max_size = 10000):
    
        self.df = df
        self.transform = transform
        
        #get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
        
        
        ##VOCAB class has been created above
        #Initialize source vocab object and build vocabulary
        self.source_vocab = Vocabulary(freq_threshold, source_vocab_max_size)
        self.source_vocab.build_vocabulary(self.source_texts.tolist())
        #Initialize target vocab object and build vocabulary
        self.target_vocab = Vocabulary(freq_threshold, target_vocab_max_size)
        self.target_vocab.build_vocabulary(self.target_texts.tolist())
        
    def __len__(self):
        return len(self.df)
    
    '''
    __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
    target values using the vocabulary objects we created in __init__
    '''
    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]
        
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        #numericalize texts ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.source_vocab.numericalize(source_text)
        numerialized_source.append(self.source_vocab.stoi["<EOS>"])
    
        numerialized_target = [self.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.target_vocab.numericalize(target_text)
        numerialized_target.append(self.target_vocab.stoi["<EOS>"])
        
        #convert the list to tensor and return
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target) 



class Validation_Dataset:
    def __init__(self, train_dataset, df, source_column, target_column, transform = None):
        self.df = df
        self.transform = transform
        
        #train dataset will be used as lookup for vocab
        self.train_dataset = train_dataset
        
        #get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        source_text = self.source_texts[index]
        #print(source_text)
        target_text = self.target_texts[index]
        #print(target_text)
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        #numericalize texts ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.train_dataset.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.train_dataset.source_vocab.numericalize(source_text)
        numerialized_source.append(self.train_dataset.source_vocab.stoi["<EOS>"])
    
        numerialized_target = [self.train_dataset.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.train_dataset.target_vocab.numericalize(target_text)
        numerialized_target.append(self.train_dataset.target_vocab.stoi["<EOS>"])
        #print(numerialized_source)
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target) 




'''
class to add padding to the batches
collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class
is used on single example
'''

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx) 
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=False, padding_value = self.pad_idx)
        return source, target




# If we run a next(iter(data_loader)) we get an output of batch_size * (num_workers+1)
def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=False):
    #get pad_idx for collate fn
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader

def get_valid_loader(dataset, train_dataset, batch_size, num_workers=1, shuffle=True, pin_memory=False):
    pad_idx = train_dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader

if __name__ == '__main__':
    client = Client()
    client
    
    datadir = 'en-fr-100.csv'

    # data = pd.read_csv('en-fr-100.csv')
    data = dd.read_csv(datadir, blocksize=24e7)

    data = data.reset_index(drop=True)
    ipdb.set_trace()
    # data.drop('source',axis=1,inplace=True)


    #preprocess


    data = data.dropna()
    #lower and remove quotes
    data['en'] = data.en.apply(lambda x: re.sub("'", '',x).lower())
    data['fr'] = data.fr.apply(lambda x: re.sub("'", '', x).lower())

    #remove special charsdata[0]
    exclude = set(string.punctuation)#set of all special chars
    #remove all the special chars
    data['en'] = data.en.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    data['fr'] = data.fr.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        
    remove_digits = str.maketrans('','',digits)
    data['en'] = data.en.apply(lambda x: x.translate(remove_digits))
    data['fr'] = data.fr.apply(lambda x: x.translate(remove_digits))

    data['fr'] = data.fr.apply(lambda x: re.sub("[२३०८१५७९४६]","",x))

    # Remove extra spaces
    data['en']=data['en'].apply(lambda x: x.strip())
    data['fr']=data['fr'].apply(lambda x: x.strip())
    data['en']=data['en'].apply(lambda x: re.sub(" +", " ", x))
    data['fr']=data['fr'].apply(lambda x: re.sub(" +", " ", x))

    ipdb.set_trace()

    # datasize = 22520376
    datasize = len(data)
    val_frac = 0.1 #precentage data in val
    val_split_idx = int(datasize*val_frac) #index on which to split
    data_idx = list(range(datasize)) #create a list of ints till len of data
    np.random.shuffle(data_idx)

    #get indexes for validation and train
    val_idx, train_idx = data_idx[:val_split_idx], data_idx[val_split_idx:]
    print('len of train: ', len(train_idx))
    print('len of val: ', len(val_idx))

    #create the sets
    train = data.iloc[train_idx].reset_index().drop('index',axis=1)
    val = data.iloc[val_idx].reset_index().drop('index',axis=1)





    train_dataset = Train_Dataset(train, 'en', 'fr')
    train_loader = get_train_loader(train_dataset, 32)
    source = next(iter(train_loader))[0]
    target = next(iter(train_loader))[1]

    print('source: \n', source)



    # #create a vocab class with freq_threshold=0 and max_size=100
    # voc = Vocabulary(0, 100)
    # sentence_list = ['that is a cat', 'that is not a dog']
    # #build vocab
    # voc.build_vocabulary(sentence_list)

    # print('index to string: ',voc.itos)
    # print('string to index:',voc.stoi)

    # print('numericalize -> cat and a dog: ', voc.numericalize('cat and a dog'))

    ipdb.set_trace()