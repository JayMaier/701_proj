'''
This file serves as a testing script for a Transformer model
'''

import torch
import ipdb
import torchtext.transforms as T
import spacy

from torch import Tensor
import math

import torch.nn as nn
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device('cpu')

en_vocab = torch.load('../models/en_vocab_500_clean.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('../models/fr_vocab_500_clean.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

device = torch.device('cpu')

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

# def save_checkpoint(state, filename="../models/2k_transv2.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)

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


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == en_vocab.get_stoi()['<PAD>']).transpose(0, 1)
    tgt_padding_mask = (tgt == en_vocab.get_stoi()['<PAD>']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device, eos_idx):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (model.transformer.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys

if __name__ == '__main__':
    batch_size = 1
    file_path = '../data/1ktest.csv'
    # file_path = '../data/en-fr-1000.csv'
    data_pipe = get_data_pipe(file_path, batch_size, 5)
    

    # Model hyperparameters
    load_model = False
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
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    # model = Transformer(
    #     embedding_size,
    #     src_vocab_size,
    #     trg_vocab_size,
    #     src_pad_idx,
    #     num_heads,
    #     num_encoder_layers,
    #     num_decoder_layers,
    #     forward_expansion,
    #     dropout,
    #     max_len,
    #     device,
    # ).to(device)
    
    learning_rate = 3e-4
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = fr_vocab.get_stoi()['<PAD>']
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    

    load_checkpoint(torch.load('../models/2k_transv2.pth.tar', map_location=torch.device('cpu')), transformer, optimizer)

    transformer.eval()

    test_bleus = []
    # ipdb.set_trace()
    for sources, targets in tqdm(data_pipe, desc=f'Evaluating...'):
        inp_data = sources.T.to(device)
        # print(inp_data)
        target = targets.T.to(device)

        # with torch.no_grad():
        #     output = model(inp_data, target)
        # ipdb.set_trace()
        src_mask, _, _,_ = create_mask(inp_data, target)
        preds = greedy_decode(transformer, inp_data, src_mask, max_len, en_vocab.get_stoi()['<BOS>'], device, en_vocab.get_stoi()['<EOS>'])
        
        pred_sen, targ_sen = [], []
        for i in range(min(inp_data.shape[0], preds.shape[0])):
            pred_sen.append(fr_vocab.get_itos()[preds[i]])
            targ_sen.append(fr_vocab.get_itos()[inp_data[i]])
        # pred_translations, target_translations, bleu = evaluate_batch(output, fr_vocab, target)
        print(pred_sen, "\n", targ_sen, '\n\n\n')
        print(bleu_score(pred_sen, targ_sen))
        # for idx, sentence in enumerate(pred_translations):
        #     print(f"Predicted translation is: \n{pred_translations[idx]}")
        #     print(f"Target translation is: \n{target_translations[idx]}")
        # print(f"Predicted translations are:\n {pred_translations}")
        # print(f"Target translations are:\n {target_translations}")

        # test_bleus.append(bleu)
    
    avg_test_bleu = sum(test_bleus) / len(data_pipe)
    print(f'The Average Bleu Score across all test batches is {avg_test_bleu}')
    # print(f"Average training loss from each epoch is: {training_losses}")
    
    
    ipdb.set_trace()