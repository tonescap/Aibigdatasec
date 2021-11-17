import os
import torch
from torch import optim
from torch import nn
from omegaconf import OmegaConf
from models.models import CharCNN
from datasets.dataset import PowershellCharDataset
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import re
import sys

def f1_score(tp, tn, total_p, total_n):
    return tp / (tp + (total_p-tp+total_n-tn)/2)

def validation(model, criterion, val_loader, writer, epoch, device, tfns):
    train_tot, train_p, train_n, val_tot, val_p, val_n = tfns
    total_loss = 0.0
    tp, tn = 0, 0
    model.eval()

    for input_batch, y_batch in val_loader:
        input_batch = input_batch.to(device)
        y_batch = y_batch.view(-1, 1).to(device)
        y_pred = model(input_batch)
        loss = criterion(y_pred, y_batch)
        total_loss += loss.item()
        tp += (torch.logical_and((y_pred>=0),(y_batch==1))).sum()
        tn += (torch.logical_and((y_pred<0),(y_batch==0))).sum()
    
    total_loss*=(train_tot/val_tot)
    acc = (tp+tn)/val_tot
    f1 = f1_score(tp, tn, val_p, val_n)
    writer.add_scalar('Loss/Val', total_loss, epoch)
    writer.add_scalar('Acc/Val', acc, epoch)
    writer.add_scalar('F1/Val', f1, epoch)
    print(f"[Val] #{epoch}\nTotal Loss: {total_loss}\nAccuracy: {acc}\nf1 score: {f1}")

def train(model, criterion, optimizer, writer, epochs, device, train_loader, val_loader, tfns, model_dir, run_name):
    train_tot, train_p, train_n, val_tot, val_p, val_n = tfns
    
    for epoch in range(0, epochs):
        total_loss = 0.0
        tp, tn = 0, 0
        model.train()
        
        for input_batch, y_batch in train_loader:
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            y_batch = y_batch.view(-1, 1).to(device)
            y_pred = model(input_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tp += (torch.logical_and((y_pred>=0),(y_batch==1))).sum()
            tn += (torch.logical_and((y_pred<0),(y_batch==0))).sum()

        acc = (tp+tn)/train_tot
        f1 = f1_score(tp, tn, train_p, train_n)
        writer.add_scalar('Loss/Train', total_loss, epoch)
        writer.add_scalar('Acc/Train', acc, epoch)
        writer.add_scalar('F1/Train', f1, epoch)

        print(f"[Train] #{epoch}\nTotal Loss: {total_loss}\nAccuracy: {acc}\nf1 score: {f1}")
        validation(model, criterion, val_loader, writer, epoch, device, tfns)

        torch.save(model.state_dict(), os.path.join(model_dir, run_name, f'{epoch}.pt'))

def normalize(document):
    document = list(document)
    for j in range(len(document)):
        if document[j] in '0123456789':
            document[j] = '*'
    return ('').join(document)

def main(cfg):
    with open('data/train/decode5/e7b28ee5-ee7d-4ce9-8e80-42bb5ce07971', 'r') as f:
        document = f.read()

    with open('data_p/token-train_decoded-2000.pkl', 'rb') as handle:
        datas = pickle.load(handle)

    sentence = re.sub("[^A-Za-z\*\$\-]+", ' ', document).lower().split()
    print(sentence)
    onehot = [datas['token_to_idx'][word] for word in sentence if word in datas['token_to_idx']]
    onehot = torch.tensor(onehot)
    print(onehot.shape)
    sys.exit()

    device = 'cpu'
    model = TokenCNN()
    checkpoint = torch.load('models/Token-CNN-W2V-0.0001-1634882115.0473955/1200.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)

if __name__ == '__main__':
    cfg = OmegaConf.load('config/train.yaml')
    main(cfg)