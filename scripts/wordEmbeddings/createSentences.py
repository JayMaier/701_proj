import pandas as pd

# Iterate through the entire dataset and split it into smaller subsets.
# This will facilitate the execution of word embeddings training in an efficient way.

i = 0
for df in pd.read_csv("archive/en-fr.csv", chunksize= 10 ** 6):
    df.dropna(inplace=True)
    # Remove punctuation marks and split sentences to words:
    df['en'] = df['en'].str.lower().str.replace("[^a-zàâçéèêîôûù]+", " ", regex=True).str.strip() + "\n"
    # Delete rows that contain only one word
    dataEn = df[df['en'].apply(lambda x: len(x.split())) > 1]['en']
    # dataFr = df[df['fr'].apply(lambda x: len(x)) > 1]['fr']
    with open("sentencesEn/sentencesEn"+str(i)+".txt", 'a') as f:
        f.write(' '.join(dataEn))
    i+=1

i = 0
for df in pd.read_csv("archive/en-fr.csv", chunksize= 10 ** 6):
    df.dropna(inplace=True)
    # Remove punctuation marks and split sentences to words:
    df['fr'] = df['fr'].str.lower().str.replace("[^a-zàâçéèêîôûù]+", " ", regex=True).str.strip() + "\n"
    # Delete rows that contain only one word
    dataEn = df[df['fr'].apply(lambda x: len(x.split())) > 1]['fr']
    with open("sentencesFr/sentencesFr"+str(i)+".txt", 'a') as f:
        f.write(' '.join(dataEn))
    i+=1