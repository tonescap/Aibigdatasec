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

def main(folder_path, checkpoint, output):
    documents = {'sentence': [], 'filename' : []}
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename),mode='r') as f:
            document = f.read()
        document = normalize(document)
        sentence = re.sub("[^A-Za-z\*\$\-]+", ' ', document).lower().split()
        documents['sentence'].append(sentence)
        documents['filename'].append(filename)

    with open('Temp.pkl', 'rb') as handle:
        datas = pickle.load(handle)

    n = torch.zeros([len(documents['sentence']), 2000], dtype = torch.int64)
    for idx, sentence in enumerate(documents['sentence']):
        onehot = [datas[word] for word in sentence if word in datas][:2000]
        n[idx, :len(onehot)] = torch.LongTensor(onehot)

    device = 'cpu'
    model = TokenCNN()
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()
    f = open(output,mode='w')
    for i in range(n.shape[0]):
        out = model(n[i].unsqueeze(0))
        if out>=0:
            f.write(documents['filename'][i] + ',Malicious\n')
        else:
            f.write(documents['filename'][i] + ',Benign\n')
    f.close()

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-o', '--output_path', required = True)
    args = parser.parse_args()

    main(args.input_dir, args.checkpoint, args.output_path)

#'Classifier/data/train/decode5/e7b28ee5-ee7d-4ce9-8e80-42bb5ce07971'
