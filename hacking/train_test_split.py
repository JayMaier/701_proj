import numpy as np
import pandas as pd
import ipdb
from tqdm import tqdm
fname = '../data/archive/en-fr_temp.csv'

# test_data = pd.read_csv(fname, skiprows = 20000000, nrows = 1000000)
# ipdb.set_trace()
# test_data.to_csv('full_test.csv', index = False)

import os
with open('tmp.csv','w') as tmp:

    with open(fname,'r') as infile:
        for linenumber, line in tqdm(enumerate(infile)):
            if linenumber < 20000000:
                tmp.write(line)