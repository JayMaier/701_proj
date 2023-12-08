from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import ipdb


direc = 'output/'




trans_raw_batch = 256 
trans_rdrop_batch = 128
lstm_embed_batch = 128
lstm_raw_batch = 128
batch_8_bit = 4
batch_4_bit = 2





fname_trans_rdrop = 'trans_rdrop_1.csv'
fname_trans_raw1 = 'trans_2epoch.csv'
fname_trans_raw2 = 'trans_2epoch_more_training.csv'

fname_lstm_embed = 'lstm_dec2_with_gen2sim.csv'
fname_lstm_raw = 'lstm_run_17hrs_dec1.csv'

fname_8_bit = 'solar-firebrand-9_lora_8_bit.csv'
fname_4_bit = 'stellar-cloud-1_lora_4_bit.csv'

df_trans_rdrop = pd.read_csv(direc + fname_trans_rdrop)
df_trans_raw1 = pd.read_csv(direc + fname_trans_raw1)
df_trans_raw2 = pd.read_csv(direc + fname_trans_raw2)
df_trans_raw2['Step'] += df_trans_raw1['Step'].iloc[-1]

df_trans_raw = pd.concat([df_trans_raw1, df_trans_raw2], axis=0, ignore_index=True)



df_lstm_embed = pd.read_csv(direc + fname_lstm_embed)
df_lstm_raw = pd.read_csv(direc + fname_lstm_raw)
df_8_bit = pd.read_csv(direc + fname_8_bit)
df_4_bit = pd.read_csv(direc + fname_4_bit)

trans_raw_start = df_trans_raw['Wall time'].iloc[0]
trans_rdrop_start = df_trans_rdrop['Wall time'].iloc[0]
lstm_embed_start = df_lstm_embed['Wall time'].iloc[0]
lstm_raw_start = df_lstm_raw['Wall time'].iloc[0]




window = 50
# ipdb.set_trace()
fig = plt.figure()
plt.title('Training Loss WRT sample count')
plt.scatter(df_trans_raw['Step'].rolling(window).mean()*trans_raw_batch, df_trans_raw['Value'].rolling(window).mean(), label = 'Vanilla Transformer')
plt.scatter(df_trans_rdrop['Step'].rolling(window).mean()*trans_rdrop_batch, df_trans_rdrop['Value'].rolling(window).mean(), label = 'Transformer with R-Drop')

plt.scatter(df_lstm_raw['Step'].rolling(window).mean()*lstm_raw_batch, df_lstm_raw['Value'].rolling(window).mean(), label = 'Vanilla LSTM')
plt.scatter(df_lstm_embed['Step'].rolling(window).mean()*lstm_embed_batch, df_lstm_embed['Value'].rolling(window).mean(), label = 'LSTM with pretrained word2vec-style embeddings (custom implementation)')

plt.scatter(df_8_bit['Step'].rolling(window).mean()*batch_8_bit, df_8_bit['solar-firebrand-9 - train/loss'].rolling(window).mean(), label = 'LLAMA2 Finetuned with LoRA (8-bit)')
plt.scatter(df_4_bit['Step'].rolling(window).mean()*batch_4_bit, df_4_bit['stellar-cloud-1 - train/loss'].rolling(window).mean(), label = 'LLAMA2 Finetuned with LoRA (4-bit)')
plt.legend()
plt.xlabel('Training Sample Count')
plt.ylabel('Training Loss (Cross Entropy)')
# plt.show()

fig1 = plt.figure()
plt.title('Training Loss WRT Training Time')
plt.scatter(df_trans_raw['Wall time'].rolling(window).mean() - trans_raw_start, df_trans_raw['Value'].rolling(window).mean(), label = 'Vanilla Transformer')
plt.scatter(df_trans_rdrop['Wall time'].rolling(window).mean() - trans_rdrop_start, df_trans_rdrop['Value'].rolling(window).mean(), label = 'Transformer with R-Drop')

plt.scatter(df_lstm_raw['Wall time'].rolling(window).mean() - lstm_raw_start, df_lstm_raw['Value'].rolling(window).mean(), label = 'Vanilla LSTM')
plt.scatter(df_lstm_embed['Wall time'].rolling(window).mean() - lstm_embed_start, df_lstm_embed['Value'].rolling(window).mean(), label = 'LSTM with pretrained word2vec-style embeddings (custom implementation)')

plt.scatter(df_8_bit['Step'].rolling(window).mean()*1.205*3, df_8_bit['solar-firebrand-9 - train/loss'].rolling(window).mean(), label = 'LLAMA2 Finetuned with LoRA (8-bit)')
plt.scatter(df_4_bit['Step'].rolling(window).mean()*1.017*3, df_4_bit['stellar-cloud-1 - train/loss'].rolling(window).mean(), label = 'LLAMA2 Finetuned with LoRA (4-bit)')
plt.legend()
plt.xlabel('Training Time (Seconds)')
plt.ylabel('Training Loss (Cross Entropy)')
plt.show()

ipdb.set_trace()