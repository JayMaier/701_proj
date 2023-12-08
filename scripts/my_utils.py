'''
Shared utility functions across models. This module includes
functions for data preprocessing, model saving and loading, 
generating masks for the Transformer, and evaluating model output
'''

import torch
import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
import torch.nn.functional as F

from torchtext.data.metrics import bleu_score

en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

### Data cleaning/prepocessing functions

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

def applyTransform(pair, en_vocab, fr_vocab):
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


def showSomeTransformedSentences(data_pipe, source_index_to_string, target_index_to_string):
    """
    Function to show how the sentences look like after applying all transforms.
    Here we try to print actual words instead of corresponding index
    """
    for sources,targets in data_pipe:
        # Just to visualize padding of shorter sentences
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
    
def get_data_pipe(file_path, batch_size, batch_num, transform_function):
    '''
    Function to instantiate a data pipe for training and evaluating
    an LSTM or Transformer
    '''
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
    data_pipe = data_pipe.map(transform_function)
    data_pipe = data_pipe.bucketbatch(batch_size = batch_size,
                                      batch_num=batch_num,
                                      bucket_num=1,
                                      use_in_batch_shuffle=False,
                                      sort_key=sortBucket)
    data_pipe = data_pipe.map(separateSourceTarget)
    data_pipe = data_pipe.map(applyPadding)
    return data_pipe

def get_data_pipe_llama(file_path):
    '''
    Function to instantiate a data pipe for tuning and evaluating Llama
    '''
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=20*10^6, delimiter = ',', as_tuple=True)
    return data_pipe


### Model maintenance functions ###

def save_checkpoint(state, filename="../models/transformer_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

### Evaluation functions ###

def evaluate_batch(batch_output, target_vocab, batch_target, max_n=4, weights=[0.25]*4, verbose=False):
    '''
    Function to evalaute batch output from an LSTM. The output is converted into string format and 
    a bleu score is computed
    Input:
        batch_output - batch output from model
        target_vocab - target vocabulary (i.e. French vocab)
        batch_target - target batch output (i.e. ground truth translations)
        max_n - number of n grams to use for bleu score
        weights - corresponding weights for each n gram
        verbose - flag to toggle debugging print
    Output:
        Predicted translations for debugging
        Target translations for debugging
        Average bleu score from batch
    '''
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
        for token_id in range(1, predicted_words.shape[0]):
            target_token = target_vocab.get_itos()[batch_target[token_id,sentence_id]]
            pred_token = target_vocab.get_itos()[predicted_words[token_id,sentence_id]]
            if pred_token == target_vocab.get_itos()[end_idx]:
                break
            pred_sentence.append(pred_token)
            target_sentence.append(target_token)

        candidate_translations.append(pred_sentence)
        reference_translations.append(target_sentence)
        reference_corpus = [reference_translations]
        bleu += bleu_score(candidate_translations, reference_corpus, max_n, weights)
        pred_translations.append(candidate_translations)
        target_translations.append(reference_corpus)
    
    if verbose:
        print(f"Average bleu score from batch is {bleu/len(target_translations)}")

    return pred_translations, target_translations, bleu / len(target_translations)

### Transformer specific functions ###

def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, en_vocab, DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == en_vocab.get_stoi()['<PAD>']).transpose(0, 1)
    tgt_padding_mask = (tgt == en_vocab.get_stoi()['<PAD>']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, device, eos_idx):
    '''
    Function to decode sentence output from Transformer model
    '''
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

def compute_kl_loss(p, q, batch_size, pad_mask=None):
    '''
    Compute average KL divergence from each sentence in batch output

    p: Output logits from one forward pass using dropout
    q: Output logits from same input using a different forward pass using dropout
    pad_mask: target padding mask
    '''
    # compute bidirectional KL divergence
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # zero out padded words
    if pad_mask is not None:
        pad_mask = pad_mask.reshape(-1).unsqueeze(dim=1)
        p_loss.masked_fill_(pad_mask.expand(-1, p_loss.shape[-1]), 0.)
        q_loss.masked_fill_(pad_mask.expand(-1, p_loss.shape[-1]), 0.)

    p_loss = p_loss.sum()/batch_size
    q_loss = q_loss.sum()/batch_size

    loss = (p_loss + q_loss)/2
    return loss