import pandas as pd

src_pth = '../data/archive/en-fr.csv'

df = pd.read_csv(src_pth, nrows = 1000000)
df.to_csv('en-fr-1000000.csv', index = False)