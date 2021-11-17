import os
import torch
from torch import optim
from torch import nn
from omegaconf import OmegaConf
from Classifier.models.models import CharCNN, TokenCNN
from Classifier.data_loader.datasets import PowershellCharDataset
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import re
import sys
import argparse

def normalize(document):
    document = list(document)
    for j in range(len(document)):
        if document[j] in '0123456789':
            document[j] = '*'
    return ('').join(document)

def main(filepath, checkpoint):
    
    with open(filepath, 'r') as f:
        document = f.read()

    with open('Temp.pkl', 'rb') as handle:
        datas = pickle.load(handle)

    document = normalize(document)
    sentence = re.sub("[^A-Za-z\*\$\-]+", ' ', document).lower().split()
    n = torch.zeros(2000, dtype = torch.int64)
    onehot = [datas[word] for word in sentence if word in datas][:2000]
    n[:len(onehot)] = torch.LongTensor(onehot)

    device = 'cpu'
    model = TokenCNN()
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()
    out = model(n.unsqueeze(0))
    if out>=0:
        print('Malicious')
    else:
        print('Benign')

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-c', '--checkpoint', required=True)
    args = parser.parse_args()
    print(args)

    main(args.filename, args.checkpoint)

#'Classifier/data/train/decode5/e7b28ee5-ee7d-4ce9-8e80-42bb5ce07971'