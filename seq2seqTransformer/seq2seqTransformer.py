'''
This program translates english to french using a seq2seq approach where
the encoder and decoder are part of a vanilla transfomer architecture
'''

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator, TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

### Helper functions ###

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def evaluate_batch(batch_output, target_vocab, batch_target):
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

### Preprocessing ###

nrows = 5000
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
    

### Setup Training Phase ###

load_model = False
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
save_model = True

# Training hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(english.vocab)
trg_vocab_size = len(french.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
### Note: must adjust this parameter to fit the longest sentence. Either throw out sentences longer than 
### max length or increase max length to accomodate the longest sentence
max_len = 100 
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard
writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.English),
    device=device,
)

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

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = french.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.ptar"), model, optimizer)


# training_losses = []
# for epoch in range(1, num_epochs+1):

#     if save_model:
#         checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
#         save_checkpoint(checkpoint)

#     loss_list = []
#     for batch in tqdm(train_iterator, desc=f'Train Epoch: {epoch}/{num_epochs}'):
#         inp_data = batch.English.to(device)
#         target = batch.French.to(device)
#         # target shape: (seq_length, batch_size)

#         # forward prop
#         output = model(inp_data, target[:-1])
#         # output shape: (seq_length-1, batch_size, vocab_size)

#         output = output.reshape(-1, output.shape[2])
#         target = target[1:].reshape(-1)
#         optimizer.zero_grad()

#         loss = criterion(output, target)
#         loss.backward()
#         loss_list.append(loss.item())

#         torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

#         optimizer.step()

#         writer.add_scalar("Training Loss", loss, global_step=step)

#     training_losses.append(sum(loss_list) / len(train_iterator))

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
    print("Batch start: \n\n")
    for idx, sentence in enumerate(pred_translations):
        print(f"Predicted translation is: \n{pred_translations[idx]}")
        print(f"Target translation is: \n{target_translations[idx]}")
    # print(f"Predicted translations are:\n {pred_translations}")
    # print(f"Target translations are:\n {target_translations}")

    test_bleus.append(bleu)
 
avg_test_bleu = sum(test_bleus) / len(test_iterator)
print(f'The Average Bleu Score across all test batches is {avg_test_bleu}')
# print(f"Average training loss from each epoch is: {training_losses}")