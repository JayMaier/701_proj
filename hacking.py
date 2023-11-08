from nltk.stem.snowball import FrenchStemmer
import ipdb
import pandas as pd


data_pth = 'data/archive/en-fr.csv'

chunk = pd.read_csv(data_pth, chunksize=100)
df = pd.concat(chunk)
print(df)
print(chunk)
ipdb.set_trace()
stemmer = FrenchStemmer()
tokens = word_tokenize(file)
french_stem = [stemmer.stem(word) for word in word_tokenize(file)]
stemmed_text = ' '.join([stemmer.stem(word) for word in word_tokenize(file)])
print(stemmed_text)