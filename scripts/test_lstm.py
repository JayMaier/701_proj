'''
This file serves as a testing script for evaluating an LSTM model
'''

import torch
import ipdb
import spacy
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model_classes import my_LSTM as LSTM
import my_utils as ut
from functools import partial

en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

device = torch.device('mps')

if __name__ == '__main__':

    # file_path = '../../data/en-fr.csv'
    file_path = '../en-fr-100.csv'
    transform_function = partial(ut.applyTransform, en_vocab=en_vocab, fr_vocab=fr_vocab)
    data_pipe = ut.get_data_pipe(file_path, 1, 5, transform_function)
    
    
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
    
    
    encoder_net = LSTM.Encoder(input_size_encoder, None, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    decoder_net = LSTM.Decoder(input_size_decoder, None, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    
    model = LSTM.Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = fr_vocab.get_stoi()['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
### Evaluating the Model ###

# Load the model for evaluation
ut.load_checkpoint(torch.load('../models/lstm_first_pass_31k.pth.tar', map_location=torch.device('mps')), model, optimizer)

model.eval()
verbose = True

test_bleus = []
for sources, targets in tqdm(data_pipe, desc=f'Evaluating:'):
    inp_data = sources.T.to(device)
    target = targets.T.to(device)
    with torch.no_grad():
        output = model.forward(inp_data, target, teacher_force_ratio=0)
    
    pred_translations, target_translations, bleu = ut.evaluate_batch(output, fr_vocab, target, max_n=1, weights=[1])
    if verbose:
        print(f"Predicted translations are:\n {pred_translations}")
        print(f"Target translations are:\n {target_translations}")

    test_bleus.append(bleu)
 
avg_test_bleu = sum(test_bleus) / len(list(data_pipe))
print(f'The Average Bleu Score across all test batches is {avg_test_bleu}')