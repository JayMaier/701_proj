import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchdata.datapipes as dp
import pandas as pd
import ipdb
import torchtext.transforms as T
import spacy

import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter

from torchtext.vocab import Vocab
from collections import Counter

en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

device = torch.device('cuda')

def getTransform(vocab):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.Truncate(30),
        T.AddToken(1, begin=True),
        T.AddToken(2, begin=False)
    )
    return text_transform

def enTokenize(text):
    return [token.text for token in en_spacy.tokenizer(text)]

def frTokenize(text):
    return [token.text for token in fr_spacy.tokenizer(text)]

def applyTransform(pair):
    return (
        getTransform(en_vocab)(enTokenize(en_clean(pair[0]))),
        getTransform(fr_vocab)(frTokenize(fr_clean(pair[1])))
    )
    
def en_clean(text):
    return text.lower().replace("[^a-z]+", " ").strip()

def fr_clean(text):
    return text.lower().replace("[^a-zàâçéèêîôûù]+", " ").strip()
    
def sortBucket(bucket):
    """
    Function to sort a given bucket. Here, we want to sort based on the length of
    source and target sequence.
    """
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))

def separateSourceTarget(sequence_pairs):
    """
    input of form: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
    output of form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`
    """
    sources,targets = zip(*sequence_pairs)
    return sources,targets

def applyPadding(pair_of_sequences):
    """
    Convert sequences to tensors and apply padding
    """
    return (T.ToTensor(3)(list(pair_of_sequences[0])), T.ToTensor(3)(list(pair_of_sequences[1])))

source_index_to_string = en_vocab.get_itos()
target_index_to_string = fr_vocab.get_itos()

def showSomeTransformedSentences(data_pipe):
    """
    Function to show how the sentences look like after applying all transforms.
    Here we try to print actual words instead of corresponding index
    """
    for sources,targets in data_pipe:
        # ipdb.set_trace()
        # if sources[0][-1] != 3:
        #     continue # Just to visualize padding of shorter sentences
        for i in range(4):
            source = ""
            for token in sources[i]:
                source += " " + source_index_to_string[token]
            target = ""
            for token in targets[i]:
                target += " " + target_index_to_string[token]
            print(f"Source: {source}")
            print(f"Traget: {target}")
        break
    
def get_data_pipe(file_path, batch_size, batch_num):
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
    data_pipe = data_pipe.map(applyTransform)
    data_pipe = data_pipe.bucketbatch(batch_size = batch_size,
                                      batch_num=batch_num,
                                      bucket_num=1,
                                      use_in_batch_shuffle=False,
                                      sort_key=sortBucket)
    data_pipe = data_pipe.map(separateSourceTarget)
    data_pipe = data_pipe.map(applyPadding)
    return data_pipe


### Helper functions

def save_checkpoint(state, filename="../models/transformer_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def evaluate_batch(batch_output, target_vocab, batch_target):
    # output shape: (trg_len, batch_size, output_dim)
    predicted_words = batch_output.argmax(2)

    end_idx = target_vocab.get_stoi()['<EOS>']

    pred_translations = []
    target_translations = []
    bleu = 0
    for sentence_id in range(predicted_words.shape[1]):
        pred_sentence = []
        target_sentence = []
        for token_id in range(predicted_words.shape[0]):
            pred_token = target_vocab.get_itos()[predicted_words[token_id,sentence_id]]
            if pred_token == target_vocab.get_itos()[end_idx] or token_id == predicted_words.shape[0]-1:
                break
            target_token = target_vocab.get_itos()[batch_target[token_id+1,sentence_id]]
            pred_sentence.append(pred_token)
            target_sentence.append(target_token)

        bleu += bleu_score(pred_sentence, target_sentence)
        pred_translations.append(pred_sentence)
        target_translations.append(target_sentence)

    print(f"Average bleu score from batch is {bleu/len(target_translations)}")

    return pred_translations, target_translations, bleu / len(target_translations) 

class Transformer(nn.Module):
    def __init__(
            self,
            embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src shape: (src_len, N)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask
    
    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask = trg_mask
        )
        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    batch_size = 2
    file_path = '../data/1ktest.csv'
    # file_path = '../data/en-fr-1000.csv'
    data_pipe = get_data_pipe(file_path, batch_size, 5)
    

    # Model hyperparameters
    load_model = False
    device = torch.device('cpu')

    


    # Model hyperparameters
    src_vocab_size = len(en_vocab)
    trg_vocab_size = len(fr_vocab)
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    ### Note: must adjust this parameter to fit the longest sentence. Either throw out sentences longer than 
    ### max length or increase max length to accomodate the longest sentence
    max_len = 32
    forward_expansion = 4
    src_pad_idx = en_vocab.get_stoi()["<PAD>"]

    model = Transformer(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ).to(device)
    
    learning_rate = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = fr_vocab.get_stoi()['<PAD>']


    load_checkpoint(torch.load('../models/16000it.pth.tar', map_location=torch.device('cpu')), model, optimizer)

    model.eval()

    test_bleus = []
    ipdb.set_trace()
    for sources, targets in tqdm(data_pipe, desc=f'Evaluating...'):
        inp_data = sources.T.to(device)
        # print(inp_data)
        target = targets.T.to(device)

        with torch.no_grad():
            output = model(inp_data, target)
        ipdb.set_trace()
        pred_translations, target_translations, bleu = evaluate_batch(output, fr_vocab, target)
        print("Batch start: \n\n")
        for idx, sentence in enumerate(pred_translations):
            print(f"Predicted translation is: \n{pred_translations[idx]}")
            print(f"Target translation is: \n{target_translations[idx]}")
        # print(f"Predicted translations are:\n {pred_translations}")
        # print(f"Target translations are:\n {target_translations}")

        test_bleus.append(bleu)
    
    avg_test_bleu = sum(test_bleus) / len(data_pipe)
    print(f'The Average Bleu Score across all test batches is {avg_test_bleu}')
    # print(f"Average training loss from each epoch is: {training_losses}")
    
    
    ipdb.set_trace()