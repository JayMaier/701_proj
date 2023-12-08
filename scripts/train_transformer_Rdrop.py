'''
This file serves as a training script for a Transformer equipped with
an additional regularization on the loss function called R-Drop
'''

import torch
import ipdb
import spacy

import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model_classes import my_Transformer as Trans
import my_utils as ut
from functools import partial

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

device = torch.device('cpu')

if __name__ == '__main__':
    batch_size = 32
    file_path = '../../data/en-fr.csv'
    # file_path = '../data/en-fr-1000.csv'
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
    alpha = 5 # R-drop regularization parameter
    
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
            # concatenate inputs along batch dimension for R-Drop
            inp_data = torch.cat((inp_data, inp_data), dim=1)
          
            target = targets.T.to(device)
            tgt_inp = target[:-1]
            
            tgt_inp = torch.cat((tgt_inp, tgt_inp), dim=1)
            tgt_out = target[1:]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = ut.create_mask(inp_data, tgt_inp, en_vocab, DEVICE)

            output = transformer(inp_data, tgt_inp, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            # output shape: (target_len, 2*batch_size, output_dim)
        
            output1 = output[:, :batch_size, :]
            output2 = output[:, batch_size:, :]

            logits1 = output1.reshape(-1, output1.shape[-1])
            logits2 = output2.reshape(-1, output2.shape[-1])

            # compute average cross entropy loss
            loss1 = loss_fn(logits1, tgt_out.reshape(-1))
            loss2 = loss_fn(logits2, tgt_out.reshape(-1))
            ce_loss = 0.5*(loss1 + loss2)
            
            # compute kl_loss
            kl_loss = ut.compute_kl_loss(logits1, logits2, batch_size, pad_mask=tgt_padding_mask[:batch_size,:].transpose(0,1))

            # add cross entropy loss and kl_loss for total loss
            loss = ce_loss + alpha*kl_loss

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