'''
This serves as a training script for an LSTM model
'''

import torch
import torchdata.datapipes as dp
import ipdb
import torchtext.transforms as T
import spacy
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

import my_utils as ut
from model_classes import my_LSTM as LSTM
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
    data_pipe = ut.get_data_pipe(file_path, 10, 5, transform_function)

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
    
    
    training_losses = []
    for epoch in range(1, num_epochs+1):

        loss_list = []
        for sources, targets in tqdm(data_pipe, desc=f'Train Epoch: {epoch}/{num_epochs}'):
            inp_data = sources.T.to(device)
            target = targets.T.to(device)

            output = model(inp_data, target)
            # output shape: (trg_len, batch_size, output_dim)
            output = output[1:].reshape(-1, output.shape[2])
            # target shape: (trg_len, batch_size)
            target = target[1:].reshape(-1)
        
            optimizer.zero_grad()
            loss = criterion(output, target)
         
            loss.backward()
            loss_list.append(loss.item())
            
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()

            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1

            if step % 100 == 0:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                ut.save_checkpoint(checkpoint)
        
        training_losses.append(sum(loss_list)/len(list(data_pipe)))
    
    print(f"Average training loss from each epoch is: {training_losses}")