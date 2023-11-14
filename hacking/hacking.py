from nltk.stem.snowball import FrenchStemmer
import ipdb
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client

# client = Client()

# client


if __name__ == '__main__':
    
    client = Client()

    client
    
    data_pth = 'data/archive/en-fr.csv'


    ddf = dd.read_csv(data_pth)
    
    ipdb.set_trace()

    print('hi')