import os
import pandas as pd
import numpy as np

train_csv = pd.read_csv('./csvs/train.csv')
train_csv = train_csv.rename(columns={'Filename': 'filename', 'Class': 'class'})
train_csv['document'] = np.nan
train_csv['obfus_lvl'] = 0
for filename in os.listdir(f'data/train/scripts'):
    with open(f'data/train/scripts/{filename}',mode='r') as f:
        train_csv.loc[train_csv['filename']==filename, 'document'] = f.read()
for i in range(1,6):
    for filename in os.listdir(f'data/train/decode{i}'):
        with open(f'data/train/decode{i}/{filename}',mode='r') as f:
            train_csv.loc[train_csv['filename']==filename, 'document'] = f.read()
            train_csv.loc[train_csv['filename']==filename, 'obfus_lvl'] = i

train_csv.to_csv('csvs/train_decoded_temp.csv', index=False)
nona_csv = pd.read_csv('csvs/train_decoded_temp.csv')
nona_csv.dropna().reset_index().to_csv('csvs/train_decoded.csv', index = False)