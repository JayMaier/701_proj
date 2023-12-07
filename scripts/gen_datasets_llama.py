import pandas as pd
import ipdb
import json
import random

train_size = 10000
eval_size = 2000
src_pth = 'data/archive/en-fr.csv'

df = pd.read_csv(src_pth, nrows = 100000)
# ipdb.set_trace()


df_tmp = df.loc[df['en'].str.len() < 200]
df_short = df_tmp.loc[df_tmp['fr'].str.len() < 200]

ids = random.sample(range(0, len(df_short)), train_size + eval_size)

outputname = 'data/en-fr-llama-train.jsonl'


with open(outputname, 'w') as tmp:
    for i in range(train_size):
        line = '### English: ' + str(df_short.iloc[ids[i]]['en']) + ' ### French: ' + str(df_short.iloc[ids[i]]['fr'])
        dict = {"text" : line}
        json.dump(dict, tmp)
        tmp.write('\n')
        
outputname = 'data/en-fr-llama-test.jsonl'


with open(outputname, 'w') as tmp:
    for i in range(train_size, train_size+eval_size):
        line = '### English: ' + str(df_short.iloc[ids[i]]['en']) + ' ### French: ' + str(df_short.iloc[ids[i]]['fr'])
        dict = {"text" : line}
        json.dump(dict, tmp)
        tmp.write('\n')
        