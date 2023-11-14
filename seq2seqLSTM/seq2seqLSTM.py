'''
This program translates english to french using a seq2seq approach where
the encoder is a vanilla LSTM and the decoder is a vanilla LSTM.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
import torchtext
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
import ipdb

### Helper functions ###

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

    end_idx = target_vocab.stoi['<eos>']

    pred_translations = []
    target_translations = []
    bleu = 0
    for sentence_id in range(predicted_words.shape[1]):
        pred_sentence = []
        target_sentence = []
        for token_id in range(predicted_words.shape[0]):
            pred_token = target_vocab.itos[predicted_words[token_id,sentence_id]]
            if pred_token == target_vocab.itos[end_idx] or token_id == predicted_words.shape[0]-1:
                break
            target_token = target_vocab.itos[batch_target[token_id+1,sentence_id]]
            pred_sentence.append(pred_token)
            target_sentence.append(target_token)

        bleu += bleu_score(pred_sentence, target_sentence)
        pred_translations.append(pred_sentence)
        target_translations.append(target_sentence)

    print(f"Average bleu score from batch is {bleu/len(target_translations)}")

    return pred_translations, target_translations, bleu / len(target_translations) 

### Preprocessing for seq2seq ###

nrows = 20000
df = pd.read_csv("../../data/en-fr.csv", nrows=nrows)

train, test = train_test_split(df, test_size=0.1)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

french = Field(sequential=True, tokenize=tokenizer_ger, use_vocab=True, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(sequential=True, tokenize=tokenizer_en, use_vocab=True, lower=True, init_token='<sos>', eos_token='<eos>')

fields = {'en': ('English', english), 'fr': ('French', french)}

train_data, test_data = TabularDataset.splits(
    path='',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=fields)

english.build_vocab(train_data, max_size=10000, min_freq=2)
french.build_vocab(train_data, max_size=10000, min_freq=2)

### Define the Architecture ###

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
        target_vocab_size = len(french.vocab)

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


### Training the Model ###

# Training hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 32

# Model hyperparameters
load_model = False
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
input_size_encoder = len(english.vocab)
input_size_decoder = len(french.vocab)
output_size = len(french.vocab)
encoder_embedding_size = 200
decoder_embedding_size = 200
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5


# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.English),
    device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = french.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

training_losses = []
for epoch in range(1, num_epochs+1):

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    loss_list = []
    # for batch_idx, batch in enumerate(train_iterator):
    for batch in tqdm(train_iterator, desc=f'Train Epoch: {epoch}/{num_epochs}'):
        inp_data = batch.English.to(device)
        target = batch.French.to(device)
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

        writer.add_scalar('Training Loss', loss, global_step=step)
        step += 1
    
    training_losses.append(sum(loss_list)/len(train_iterator))

### Evaluating the Model ###

# Save current model state and load it for evaluation
checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
save_checkpoint(checkpoint)
load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

model.eval()

test_bleus = []
for batch in tqdm(test_iterator, desc=f'Evaluating:'):
    inp_data = batch.English.to(device)
    target = batch.French.to(device)
    with torch.no_grad():
        output = model(inp_data, target)
    
    pred_translations, target_translations, bleu = evaluate_batch(output, french.vocab, target)
    # print(f"{len(pred_translations)} predicted translations of length {len(pred_translations[0])}\n")
    # print(f"{len(target_translations)} target translations of length {len(target_translations[0])}\n")
    print(f"Predicted translations are:\n {pred_translations}")
    print(f"Target translations are:\n {target_translations}")

    test_bleus.append(bleu)
 
avg_test_bleu = sum(test_bleus) / len(test_iterator)
print(f'The Average Bleu Score across all test batches is {avg_test_bleu}')
print(f"Average training loss from each epoch is: {training_losses}")









