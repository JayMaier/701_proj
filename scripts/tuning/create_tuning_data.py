import numpy as np
import pandas as pd
import random

# Create datasets for tuning experiments
# Sample 1 million lines from the first half of the training
# dataset to obtain the tuning training dataset and then sample
# 100 thousand lines from the second half of the training dataset
# to obtain the tuning validation dataset
train_size = int(0.9 * 22.5 * 10 ** 6)
half_size = 10 ** 6
tuning_train_size = 10 ** 6
tuning_test_size = 10 ** 5
train_data = np.arange(1, half_size + 1)
sample = random.choices(train_data, k=tuning_train_size)
skip_rows = list(set(train_data) - set(sample))
df = pd.read_csv("../archive/en-fr.csv", skiprows=skip_rows)
train = df[:tuning_train_size]
test = df[half_size:train_size].sample(tuning_test_size)
train.to_csv("tuning_train_data.csv", index=None)
test.to_csv("tuning_test_data.csv", index=None)