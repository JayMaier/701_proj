import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchdata.datapipes as dp
import pandas as pd
import ipdb
import torchtext.transforms as T
import spacy
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from torchtext.vocab import Vocab
from collections import Counter

en_vocab = torch.load('../models/en_vocab.pth')
en_vocab.set_default_index(en_vocab['<unk>'])
fr_vocab = torch.load('../models/fr_vocab.pth')
fr_vocab.set_default_index(fr_vocab['<unk>'])

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

device = torch.device('mps')

def getTransform(vocab):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
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
        getTransform(en_vocab)(enTokenize(pair[0])),
        getTransform(fr_vocab)(frTokenize(pair[1]))
    )
    
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

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
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
        candidate_translations = []
        reference_translations = []
        pred_sentence = []
        target_sentence = []
        for token_id in range(predicted_words.shape[0]):
            pred_token = target_vocab.get_itos()[predicted_words[token_id,sentence_id]]
            if pred_token == target_vocab.get_itos()[end_idx] or token_id == predicted_words.shape[0]-1:
                break
            target_token = target_vocab.get_itos()[batch_target[token_id+1,sentence_id]]
            pred_sentence.append(pred_token)
            target_sentence.append(target_token)

        candidate_translations.append(pred_sentence)
        reference_translations.append(target_sentence)
        reference_corpus = [reference_translations]
        bleu += bleu_score(candidate_translations, reference_corpus)
        pred_translations.append(candidate_translations)
        target_translations.append(reference_corpus)

    print(f"Average bleu score from batch is {bleu/len(target_translations)}")

    return pred_translations, target_translations, bleu / len(target_translations) 

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x: (N) but we want (1, N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, hidden_size)

        predictions = self.fc(outputs)
        # shape of predictions: (1, N, length_of_vocab)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(fr_vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)
        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs



if __name__ == '__main__':

    # file_path = '../../data/en-fr.csv'
    file_path = '../en-fr-100.csv'
    data_pipe = get_data_pipe(file_path, 1, 5)
    
    
    # showSomeTransformedSentences(data_pipe)

    # Tensorboard
    writer = SummaryWriter(f'runs/loss_plot')
    step = 0
    
    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Model hyperparameters
    load_model = False
    device = torch.device('mps')
    input_size_encoder = len(en_vocab)
    input_size_decoder = len(fr_vocab)
    output_size = len(fr_vocab)
    encoder_embedding_size = 200
    decoder_embedding_size = 200
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    
    
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    
    
    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = fr_vocab.get_stoi()['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
### Evaluating the Model ###

# Load the model for evaluation
load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

model.eval()

test_bleus = []
for sources, targets in tqdm(data_pipe, desc=f'Evaluating:'):
    inp_data = sources.T.to(device)
    target = targets.T.to(device)
    with torch.no_grad():
        output = model(inp_data, target)
    
    pred_translations, target_translations, bleu = evaluate_batch(output, fr_vocab, target)
    # print(f"{len(pred_translations)} predicted translations of length {len(pred_translations[0])}\n")
    # print(f"{len(target_translations)} target translations of length {len(target_translations[0])}\n")
    print(f"Predicted translations are:\n {pred_translations}")
    print(f"Target translations are:\n {target_translations}")

    test_bleus.append(bleu)
 
avg_test_bleu = sum(test_bleus) / len(list(data_pipe))
print(f'The Average Bleu Score across all test batches is {avg_test_bleu}')