'''
This program evaluates the performance of each LSTM and Transformer model
using the bleu score metric. The user can define how many n_grams you wish
to evaluate on by setting the max_n parameter near the top of the script to
the desired value
'''
import my_utils as ut
from model_classes import my_LSTM as LSTM
from model_classes import my_Transformer as Trans

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import spacy
from torchtext.data.metrics import bleu_score
from gensim.models import Word2Vec
from functools import partial

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import ipdb


### Prepare device ###

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('mps')

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

### Evaluation parameters ###
max_n = 3
weights = [1/max_n]*max_n
verbose = False
batch_size = 1

######################################
### LSTM No Pre-Trained Embeddings ###
######################################

### Instantiate LSTM ###

# Load appropriate dataset
en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

transform_function = partial(ut.applyTransform, en_vocab=en_vocab, fr_vocab=fr_vocab)

file_path = '../test_data_1k.csv'
data_pipe = ut.get_data_pipe(file_path, batch_size, 5, transform_function)

# Model hyperparameters
device = torch.device('mps')
input_size_encoder = len(en_vocab)
input_size_decoder = len(fr_vocab)
output_size = len(fr_vocab)
embedding_size = 200
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5
learning_rate = 0.001

encoder_net = LSTM.Encoder(input_size_encoder, None, embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = LSTM.Decoder(input_size_decoder, None, embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)


lstm1 = LSTM.Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(lstm1.parameters(), lr=learning_rate)

pad_idx = fr_vocab.get_stoi()['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

### Evaluate LSTM ###

# Load the model for evaluation
ut.load_checkpoint(torch.load('../models/lstm_first_pass_31k.pth.tar', map_location=torch.device('mps')), lstm1, optimizer)

lstm1.eval()

test_bleus = []
for sources, targets in tqdm(data_pipe, desc=f'Evaluating LSTM1:'):
    inp_data = sources.T.to(device)
    target = targets.T.to(device)
    with torch.no_grad():
        output = lstm1.forward(inp_data, target, teacher_force_ratio=0)
    
    try:
        pred_translations, target_translations, bleu = ut.evaluate_batch(output, fr_vocab, target, max_n, weights)
    except IndexError:
        print("IndexError triggered by bleu_score\n")

    if verbose:
        print(f"Predicted translations are:\n {pred_translations}")
        print(f"Target translations are:\n {target_translations}")

    test_bleus.append(bleu)
 
n = len(list(data_pipe))
avg_test_bleu_lstm1 = sum(test_bleus) / n
var_test_bleu_lstm1 = np.var(test_bleus)
z = norm.ppf(0.975)
lower_bound_lstm1 = max(0, avg_test_bleu_lstm1 - z*np.sqrt(var_test_bleu_lstm1/n))
upper_bound_lstm1 = min(1, avg_test_bleu_lstm1 + z*np.sqrt(var_test_bleu_lstm1/n))
print(f"LSTM Evaluation Summary:")
print(f'The Average Bleu Score across all test batches is {avg_test_bleu_lstm1}')
print(f"Confidence interval for average bleu score is [{lower_bound_lstm1},{upper_bound_lstm1}]\n")

###################################
### LSTM Pre-Trained Embeddings ###
###################################

### Instantiate LSTM ###

# Load appropriate dataset
en_vocab = torch.load('../lstmCustomEmbeddings/en_vocab_500.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../lstmCustomEmbeddings/fr_vocab_500.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

transform_function = partial(ut.applyTransform, en_vocab=en_vocab, fr_vocab=fr_vocab)

file_path = '../test_data_1k.csv'
data_pipe = ut.get_data_pipe(file_path, batch_size, 5, transform_function)

# Model hyperparameters
device = torch.device('mps')
input_size_encoder = len(en_vocab)
input_size_decoder = len(fr_vocab)
output_size = len(fr_vocab)
encoder_embedding_size = 200
decoder_embedding_size = 200
embedding_size = 100
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5
learning_rate = 0.001


enEmbeddings = torch.tensor(Word2Vec.load("../lstmCustomEmbeddings/embeddingsEn.model").wv.vectors)
frEmbeddings = torch.tensor(Word2Vec.load("../lstmCustomEmbeddings/embeddingsFr.model").wv.vectors)

encoder_net = LSTM.Encoder(None, enEmbeddings, embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = LSTM.Decoder(None, frEmbeddings, embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

lstm2 = LSTM.Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(lstm2.parameters(), lr=learning_rate)

pad_idx = fr_vocab.get_stoi()['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

### Evaluate LSTM ###

# Load the model for evaluation
ut.load_checkpoint(torch.load('../lstmCustomEmbeddings/my_checkpoint.pth.tar', map_location=torch.device('mps')), lstm2, optimizer)

lstm2.eval()

test_bleus = []
for sources, targets in tqdm(data_pipe, desc=f'Evaluating LSTM2:'):
    inp_data = sources.T.to(device)
    target = targets.T.to(device)
    with torch.no_grad():
        output = lstm2.forward(inp_data, target, teacher_force_ratio=0)
    
    try:
        pred_translations, target_translations, bleu = ut.evaluate_batch(output, fr_vocab, target, max_n, weights)
    except IndexError:
        print("IndexError triggered by bleu_score\n")

    if verbose:
        print(f"Predicted translations are:\n {pred_translations}")
        print(f"Target translations are:\n {target_translations}")

    test_bleus.append(bleu)
 
n = len(list(data_pipe))
avg_test_bleu_lstm2 = sum(test_bleus) / n
var_test_bleu_lstm2 = np.var(test_bleus)
z = norm.ppf(0.975)
lower_bound_lstm2 = max(0, avg_test_bleu_lstm2 - z*np.sqrt(var_test_bleu_lstm2/n))
upper_bound_lstm2 = min(1, avg_test_bleu_lstm2 + z*np.sqrt(var_test_bleu_lstm2/n))
print(f"LSTM Evaluation Summary:")
print(f'The Average Bleu Score across all test batches is {avg_test_bleu_lstm2}')
print(f"Confidence interval for average bleu score is [{lower_bound_lstm2},{upper_bound_lstm2}]\n")

#############################
### Transformer No R-Drop ###
#############################

### Instantiate Transformer ###

# Load appriopriate dataset
en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

transform_function = partial(ut.applyTransform, en_vocab=en_vocab, fr_vocab=fr_vocab)

file_path = '../test_data_1k.csv'
data_pipe = ut.get_data_pipe(file_path, batch_size, 5, transform_function)

# Model hyperparameters
device = torch.device('cpu')

SRC_VOCAB_SIZE = len(en_vocab)
TGT_VOCAB_SIZE = len(fr_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
max_len = 32

transformer = Trans.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

learning_rate = 3e-4

pad_idx = fr_vocab.get_stoi()['<PAD>']
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

### Evaluate transformer ###

ut.load_checkpoint(torch.load('../models/transformer_checkpoint_v2_nordrop_most_train.pth.tar', map_location=torch.device('mps')), transformer, optimizer)

transformer.eval()

test_bleus = []
for sources, targets in tqdm(data_pipe, desc=f'Evaluating Transformer:'):
    inp_data = sources.T.to(device)
    target = targets.T.to(device)

    src_mask, _, _,_ = ut.create_mask(inp_data, target, en_vocab, DEVICE)
    preds = ut.greedy_decode(transformer, inp_data, src_mask, max_len, en_vocab.get_stoi()['<BOS>'], device, en_vocab.get_stoi()['<EOS>'])
    
    candidate_translations, reference_translations = [], []
    pred_sen, targ_sen = [], []
    for i in range(min(inp_data.shape[0], preds.shape[0])):
        pred_sen.append(fr_vocab.get_itos()[preds[i]])
        targ_sen.append(fr_vocab.get_itos()[inp_data[i]])

    candidate_translations.append(pred_sen)
    reference_translations.append(targ_sen)
    reference_corpus = [reference_translations]
   
    if verbose:
        print(f"Predicted sentence is {pred_sen}\n")
        print(f"Target translation is {targ_sen}\n")

    try:
        bleu = bleu_score(candidate_translations, reference_corpus, max_n, weights)
    except IndexError:
        print("IndexError triggered by bleu score\n")

    test_bleus.append(bleu)

n = len(list(data_pipe))
avg_test_bleu_trans = sum(test_bleus) / n
var_test_bleu_trans = np.var(test_bleus)
z = norm.ppf(0.975)
lower_bound_trans = max(0, avg_test_bleu_trans - z*np.sqrt(var_test_bleu_trans/n))
upper_bound_trans = min(1, avg_test_bleu_trans + z*np.sqrt(var_test_bleu_trans/n))
print(f"Transformer Evaluation Summary:")
print(f'The Average Bleu Score across all test batches is {avg_test_bleu_trans}')
print(f"Confidence interval for average bleu score is [{lower_bound_trans},{upper_bound_trans}]\n")

#############################
### Transformer R-Drop ######
#############################

# Load appriopriate dataset
en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

transform_function = partial(ut.applyTransform, en_vocab=en_vocab, fr_vocab=fr_vocab)

file_path = '../test_data_1k.csv'
data_pipe = ut.get_data_pipe(file_path, batch_size, 5, transform_function)

# Model hyperparameters
device = torch.device('cpu')

SRC_VOCAB_SIZE = len(en_vocab)
TGT_VOCAB_SIZE = len(fr_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
max_len = 32

transformerR = Trans.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

learning_rate = 3e-4

pad_idx = fr_vocab.get_stoi()['<PAD>']
optimizer = torch.optim.Adam(transformerR.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

### Evaluate transformer ###

ut.load_checkpoint(torch.load('../models/transformer_checkpoint_rdrop2.pth.tar', map_location=torch.device('mps')), transformerR, optimizer)

transformerR.eval()

test_bleus = []
for sources, targets in tqdm(data_pipe, desc=f'Evaluating TransformerR:'):
    inp_data = sources.T.to(device)
    target = targets.T.to(device)

    src_mask, _, _,_ = ut.create_mask(inp_data, target, en_vocab, DEVICE)
    preds = ut.greedy_decode(transformer, inp_data, src_mask, max_len, en_vocab.get_stoi()['<BOS>'], device, en_vocab.get_stoi()['<EOS>'])
    
    candidate_translations, reference_translations = [], []
    pred_sen, targ_sen = [], []
    for i in range(min(inp_data.shape[0], preds.shape[0])):
        pred_sen.append(fr_vocab.get_itos()[preds[i]])
        targ_sen.append(fr_vocab.get_itos()[inp_data[i]])

    candidate_translations.append(pred_sen)
    reference_translations.append(targ_sen)
    reference_corpus = [reference_translations]
   
    if verbose:
        print(f"Predicted sentence is {pred_sen}\n")
        print(f"Target translation is {targ_sen}\n")

    try:
        bleu = bleu_score(candidate_translations, reference_corpus, max_n, weights)
    except IndexError:
        print("IndexError triggered by bleu score\n")

    test_bleus.append(bleu)

n = len(list(data_pipe))
avg_test_bleu_transR = sum(test_bleus) / n
var_test_bleu_transR = np.var(test_bleus)
z = norm.ppf(0.975)
lower_bound_transR = max(0, avg_test_bleu_transR - z*np.sqrt(var_test_bleu_transR/n))
upper_bound_transR = min(1, avg_test_bleu_transR + z*np.sqrt(var_test_bleu_transR/n))
print(f"Transformer Evaluation Summary:")
print(f'The Average Bleu Score across all test batches is {avg_test_bleu_transR}')
print(f"Confidence interval for average bleu score is [{lower_bound_transR},{upper_bound_transR}]\n")

###################
### Plot Output ###
###################

models = ["LSTM1","LSTM2", "Transformer", "Transformer-R"]
bleu_scores = [avg_test_bleu_lstm1, avg_test_bleu_lstm2, avg_test_bleu_trans, avg_test_bleu_transR]
positive_errors = [upper_bound_lstm1-avg_test_bleu_lstm1, upper_bound_lstm2-avg_test_bleu_lstm2, upper_bound_trans-avg_test_bleu_trans, upper_bound_transR-avg_test_bleu_transR]
negative_errors = [avg_test_bleu_lstm1-lower_bound_lstm1, avg_test_bleu_lstm2-lower_bound_lstm2, avg_test_bleu_trans-lower_bound_trans, avg_test_bleu_transR-lower_bound_transR]

plt.bar(models, bleu_scores)

plt.errorbar(models, bleu_scores, yerr=[negative_errors, positive_errors], fmt='none', ecolor='red',elinewidth=2, capsize=5)
plt.xlabel('Models')
plt.ylabel('Bleu Score')
plt.title(f'Model Evaluation with Max N-Grams={max_n}')

filename = f'model_evaluation_max_n={max_n}.png'
plt.savefig(filename)







