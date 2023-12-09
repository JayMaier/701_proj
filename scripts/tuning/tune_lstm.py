import re
import torch
import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

from gensim.models import Word2Vec

# Load vocabularies
en_vocab = torch.load('en_vocab_500.pth')
en_vocab.set_default_index(en_vocab['<UNK>'])
fr_vocab = torch.load('fr_vocab_500.pth')
fr_vocab.set_default_index(fr_vocab['<UNK>'])

# Load spacy tokenizers
spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def getTransform(vocab):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.Truncate(30),
        T.AddToken(1, begin=True),
        T.AddToken(2, begin=False)
    )
    return text_transform

def enTokenize(text):
    return [str.lower(token.text) for token in spacy_en.tokenizer(text)
            if re.fullmatch(r"[a-z'`]+", str.lower(token.text)) is not None]

def frTokenize(text):
    return [str.lower(token.text) for token in spacy_fr.tokenizer(text)
            if re.fullmatch(r"[a-zàâçéèêîôûù'`]+", str.lower(token.text)) is not None]

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
    
def get_data_pipe(file_path, batch_size, skip=False, batch_num=5, train_size=1):
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    if skip:
        data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter = ',', as_tuple=True)
    else:
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

class Encoder(nn.Module):
    def __init__(self, embeddings, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        if embeddings == None:
            self.embedding = nn.Embedding(len(en_vocab), embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, embeddings, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        if embeddings == None:
            self.embedding = nn.Embedding(len(fr_vocab), embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
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


# Train a setting on the tuning train dataset
# During training, we checkpoint 10 models accross an epoch that will be used to perform evaluation
# on the tuning validation dataset
def train(model, optimizer, train_data, batch_size, model_signature, num_epochs=1):
    train_data.reset()
    checkpoint_step = int(10 ** 5 / batch_size) # step at which a checkpoint evaluation on test dataset will be run
    print("### Training on model", model_signature, "...")
    training_losses = []
    step = 0
    writer = SummaryWriter(f'runs/' + model_signature + '/train_loss')
    for epoch in range(1, num_epochs+1):

        loss_list = []
        for sources, targets in tqdm(train_data, desc=f'Train Epoch: {epoch}/{num_epochs}'):
            inp_data = sources.T.to(device)
            target = targets.T.to(device)
            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            loss_list.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            torch.cuda.empty_cache()
            del inp_data, target

            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1
            
            if step % checkpoint_step == 0:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                i = int(step / checkpoint_step)
                save_checkpoint(checkpoint, 'models/' + model_signature + '_step_' + str(i) + '.pth.tar')
        training_losses.append(sum(loss_list)/len(loss_list))
        print("Train Loss for epoch", epoch, sum(loss_list)/len(loss_list))

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint, 'models/' + model_signature + '_step_10.pth.tar')


# Evaluate model on the entire test dataset and report test loss
def test(model, data, model_signature, writer, step):
    print("### Testing on model", model_signature, "for step " + str(step) + "...")

    loss_list = []
    with torch.no_grad():
        for sources, targets in tqdm(data, desc=f'Test Epoch: {1}/{1}'):
            inp_data = sources.T.to(device)
            target = targets.T.to(device)
            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            loss = criterion(output, target)
            loss_list.append(loss.item())
            torch.cuda.empty_cache()
            del inp_data, target

    writer.add_scalar('Test Loss', sum(loss_list)/len(loss_list), global_step=step)        
    print("Test Loss", sum(loss_list)/len(loss_list))


if __name__ == '__main__':
    # Hyper-params
    learning_rate = 0.001
    batch_sizes = [32, 64, 128]
    drop_rates = [0.2, 0.5, 0.6]
    customEmbeddings = [True, False]
    num_epochs = 5
    train_size = 10 ** 6
    test_size = 10 ** 5

    # Model hyperparameters
    device = torch.device('cuda') # CHANGE BEFORE RUNNING
    torch.cuda.set_per_process_memory_fraction(0.85)  # Set to an appropriate fraction
    input_size_encoder = len(en_vocab)
    input_size_decoder = len(fr_vocab)
    output_size = len(fr_vocab)
    embedding_size = 100
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    file_path = 'tuning_train_data.csv'
    test_file_path = 'tuning_test_data.csv'

    # Construct models - Standard version
    st_encoder_net = Encoder(None, embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    st_decoder_net = Decoder(None, embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    standard_model = Seq2Seq(st_encoder_net, st_decoder_net).to(device)
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=learning_rate)

    # Construct models - Custom version
    enEmbeddings = torch.tensor(Word2Vec.load("wordEmbeddings/embeddingsEn.model").wv.vectors)
    frEmbeddings = torch.tensor(Word2Vec.load("wordEmbeddings/embeddingsFr.model").wv.vectors)
    encoder_net = Encoder(enEmbeddings, embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    decoder_net = Decoder(frEmbeddings, embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    custom_model = Seq2Seq(encoder_net, decoder_net).to(device)
    custom_optimizer = optim.Adam(custom_model.parameters(), lr=learning_rate)

    pad_idx = fr_vocab.get_stoi()['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Create an iterable of the test dataset
    test_data = get_data_pipe(test_file_path, 1)

    # Tune batch size
    for batch_size in batch_sizes:
        data_pipe = get_data_pipe(file_path, batch_size)
        train(custom_model, custom_optimizer, data_pipe, batch_size, "model_batch_" + str(batch_size))

    # Tune Custom vs Standard
    batch_size = 128
    data_pipe = get_data_pipe(file_path, batch_size)
    train(custom_model, custom_optimizer, data_pipe, batch_size, "model_custom")
    train(standard_model, standard_optimizer, data_pipe, batch_size, "model_standard")


    # Tune dropout rate
    for dropout in drop_rates:
        encoder_net = Encoder(enEmbeddings, embedding_size, hidden_size, num_layers, dropout).to(device)
        decoder_net = Decoder(frEmbeddings, embedding_size, hidden_size, output_size, num_layers, dropout).to(device)
        custom_model = Seq2Seq(encoder_net, decoder_net).to(device)
        custom_optimizer = optim.Adam(custom_model.parameters(), lr=learning_rate)
        train(custom_model, custom_optimizer, data_pipe, batch_size, "model_drop_" + str(dropout))

    # Report test loss on checkpoints
    for model_sign in ['model_batch_32', 'model_batch_64', 'model_batch_128', 'model_custom',
                       'model_standard', 'model_drop_0.2', 'model_drop_0.5', 'model_drop_0.6']:
        model = custom_model
        optimizer = custom_optimizer
        if model_sign == 'model_standard':
            model = standard_model
            optimizer = standard_optimizer
        test_writer = SummaryWriter(f'runs/' + model_sign + '/test_loss')
        for i in range(1, 11):
            chkpt = torch.load('models/' + model_sign + '_step_' + str(i) + '.pth.tar')
            load_checkpoint(chkpt, model, optimizer)
            test(model, test_data, model_sign, test_writer, i * 10)
            test_data.reset()
