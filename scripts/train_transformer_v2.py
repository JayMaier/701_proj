'''
This file serves as a training script for a Transformer model
Note: This is the most up to date training script for a Transformer
and should be used rather than train_transformer.py
'''

import torch
import ipdb
import spacy

from model_classes import my_Transformer as Trans
from functools import partial
import my_utils as ut

import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

device = torch.device('cpu')

if __name__ == '__main__':
    batch_size = 256
    # file_path = '../data/archive/en-fr.csv'
    file_path = '../en-fr-100.csv'
    transform_function = partial(ut.applyTransform, en_vocab=en_vocab, fr_vocab=fr_vocab)
    data_pipe = ut.get_data_pipe(file_path, batch_size, 5, transform_function)

    SRC_VOCAB_SIZE = len(en_vocab)
    TGT_VOCAB_SIZE = len(fr_vocab)
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    max_len = 32

    # Model hyperparameters
    load_model = False
    device = torch.device('cpu')

    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 3e-4
    
    transformer = Trans.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_vocab.get_stoi()['<PAD>'])

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    writer = SummaryWriter(f"runs/loss_plot")
    step = 0
    
    training_losses = []

    for epoch in range(1, num_epochs+1):

        loss_list = []

        for sources, targets in tqdm(data_pipe, desc=f'Train Epoch: {epoch}/{num_epochs}'):
            inp_data = sources.T.to(device)

            target = targets.T.to(device)
            tgt_inp = target[:-1]
            tgt_out = target[1:]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = ut.create_mask(inp_data, tgt_inp, en_vocab, DEVICE)

            output = transformer(inp_data, tgt_inp, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))

            optimizer.zero_grad()

            loss.backward()
            loss_list.append(loss.item())

            optimizer.step()

            writer.add_scalar('Training Loss', loss, global_step=step)
            step += 1
            
            if step % 1000 == 0:
                checkpoint = {'state_dict': transformer.state_dict(), 'optimizer': optimizer.state_dict()}
                ut.save_checkpoint(checkpoint)
        
        training_losses.append(sum(loss_list)/len(list(data_pipe)))
    
    ipdb.set_trace()