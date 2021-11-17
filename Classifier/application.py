import os
import torch
from torch import optim
from torch import nn
from omegaconf import OmegaConf
from models.models import CharCNN, TokenCNN
from data_loader.datasets import PowershellCharDataset
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import re
import sys

def normalize(document):
    document = list(document)
    for j in range(len(document)):
        if document[j] in '0123456789':
            document[j] = '*'
    return ('').join(document)

def main(cfg):
    with open('data/train/decode5/e7b28ee5-ee7d-4ce9-8e80-42bb5ce07971', 'r') as f:
        document = f.read()

    with open('data_processed/token-train_decoded-2000.pkl', 'rb') as handle:
        datas = pickle.load(handle)

    document = normalize(document)
    sentence = re.sub("[^A-Za-z\*\$\-]+", ' ', document).lower().split()
    n = torch.zeros(2000, dtype = torch.int64)
    onehot = [datas['token_to_idx'][word] for word in sentence if word in datas['token_to_idx']][:2000]
    n[:len(onehot)] = torch.LongTensor(onehot)

    device = 'cpu'
    model = TokenCNN()
    checkpoint = torch.load('checkpoints/TokenCNN-1e-05-1637165909.085637/0.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()
    out = model(n.unsqueeze(0))
    if out>=0:
        print('Malicious')
    else:
        print('Benign')

if __name__ == '__main__':
    cfg = OmegaConf.load('config/train.yaml')
    main(cfg)