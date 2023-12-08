'''
This program evaluates the performance of each of our models
'''
# import my_utils as ut
# from model_classes import my_LSTM as LSTM
# from model_classes import my_Transformer as Trans

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# import spacy
# from torchtext.data.met/rics import bleu_score
# from gensim.models import Word2Vec
from functools import partial

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import ipdb
import pandas as pd




from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from trl import SFTTrainer, is_xpu_available
import time

# def get_bleu(en, fr):
    


base_model = "models/Llama-2-7b-hf"
device = "cuda" # or "cuda" if you have a GPU
device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
torch_dtype = torch.bfloat16

file_path = 'data/en-fr-test-1000.csv'
df = pd.read_csv(file_path, header=0, names=['en', 'fr'])
# ipdb.set_trace()

#############################################################
#4bit
print('\n###################################################\nrunning 4 bit lora\n')

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                        )

model_name_or_path = "models/merged_adapters_4bit" #path/to/your/model/or/name/on/hub


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=quantization_config, device_map=device_map, torch_dtype=torch_dtype)

tokenizer = AutoTokenizer.from_pretrained(base_model)


ens, frs, translates = [], [], []
for i in tqdm(range(1000)):
    prompt = '### English: ' + df.iloc[i]['en'] + ' ## French: '
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=2*inputs.shape[1])
    trans = tokenizer.decode(outputs[0])[len(prompt)+4:]
    print('english: \n', df.iloc[i]['en'])
    print('\n\nfrench: \n', tokenizer.decode(outputs[0]))
    ens.append(df.iloc[i]['en'])
    frs.append(df.iloc[i]['fr'])
    translates.append(trans)


output_dict = {'english': ens, 'french': frs, 'translation':translates}
out_df = pd.DataFrame(output_dict)

out_df.to_csv('llama_output_4bit_lora.csv')


#############################################################
#8bit
print('\n###################################################\nrunning 8 bit lora\n')
model_name_or_path = "models/merged_adapters_8bit"

quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, load_in_4bit=False)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=quantization_config, device_map=device_map, torch_dtype=torch_dtype)

tokenizer = AutoTokenizer.from_pretrained(base_model)


ens, frs, translates = [], [], []
for i in tqdm(range(1000)):
    prompt = '### English: ' + df.iloc[i]['en'] + ' ## French: '
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=2*inputs.shape[1])
    trans = tokenizer.decode(outputs[0])[len(prompt)+4:]
    print('english: \n', df.iloc[i]['en'])
    print('\n\nfrench: \n', tokenizer.decode(outputs[0]))
    ens.append(df.iloc[i]['en'])
    frs.append(df.iloc[i]['fr'])
    translates.append(trans)


output_dict = {'english': ens, 'french': frs, 'translation':translates}
out_df = pd.DataFrame(output_dict)

out_df.to_csv('llama_output_8bit_lora.csv')


#############################################################
#raw

print('\n###################################################\nrunning raw llama2\n')

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=None, device_map=device_map, torch_dtype=torch_dtype)

tokenizer = AutoTokenizer.from_pretrained(base_model)


ens, frs, translates = [], [], []
for i in tqdm(range(1000)):
    prompt = '### English: ' + df.iloc[i]['en'] + ' ## French: '
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=2*inputs.shape[1])
    trans = tokenizer.decode(outputs[0])[len(prompt)+4:]
    print('english: \n', df.iloc[i]['en'])
    print('\n\nfrench: \n', tokenizer.decode(outputs[0]))
    ens.append(df.iloc[i]['en'])
    frs.append(df.iloc[i]['fr'])
    translates.append(trans)


output_dict = {'english': ens, 'french': frs, 'translation':translates}
out_df = pd.DataFrame(output_dict)

out_df.to_csv('llama_output_raw.csv')


ipdb.set_trace()







