import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ipdb
import dask.dataframe as dd
from dask.distributed import Client



class CSVDataset(Dataset):
    def __init__(self, path, chunksize, data_size):
        self.path = path
        self.chunksize = chunksize
        self.len = data_size // self.chunksize

    def __getitem__(self, index):
        x = next(pd.read_csv(self.path, skiprows = index*self.chunksize, chunksize = self.chunksize))
        # ipdb.set_trace()
        # print(x)
        return(torch.from_numpy(x.values))
        
    def __len__(self):
        return self.len
    
    
if __name__ == '__main__':
    client = Client()
    
    # datadir = 'data/archive/en-fr.csv'
    datadir = 'en-fr-100.csv'
    # data_dd = dd.read_csv(datadir, blocksize = 24e7)
    # datasize = len(data_dd)
    # print(datasize)
    datasize = 22520376
    
    dataset = CSVDataset(datadir, chunksize = 1, data_size = datasize)
    loader = DataLoader(dataset, batch_size = 10, num_workers = 1, shuffle = False)
    # ipdb.set_trace()
    for batch_idx, data in enumerate(loader):
        print('once thruloader \n\n\n')
        print('batch: {}\tdata: {}'.format(batch_idx, data))
        ipdb.set_trace()