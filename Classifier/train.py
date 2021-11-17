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

def main(cfg):
    run_name = f'{cfg.model.name}-{cfg.train.lr}-{time.time()}'
    model_dict = {'CharCNN': CharCNN,
                  'TokenCNN': TokenCNN}

    with open(cfg.data.path, 'rb') as handle:
        datas = pickle.load(handle)

    with open(cfg.w2v.path, 'rb') as handle:
        w2v = pickle.load(handle)

    train_dataset = PowershellCharDataset(torch.LongTensor(datas['token_data']), torch.Tensor(datas['labels']))
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True)

    model = TokenCNN(torch.Tensor(w2v.wv.vectors)).to(cfg.train.device)
    bcell = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = cfg.train.lr)
    writer = SummaryWriter(os.path.join(cfg.output.log.dir, run_name))
    os.mkdir(os.path.join(cfg.output.model.dir, run_name))

    train_tot = len(train_dataset)
    val_tot = len(val_dataset)
    train_p = 0
    val_p = 0
    for data in train_dataset:
        if data[1]==1:
            train_p+=1
    for data in val_dataset:
        if data[1]==1:
            val_p+=1
    train_n = train_tot-train_p
    val_n = val_tot-val_p
    tfns = (train_tot, train_p, train_n, val_tot, val_p, val_n)
    writer.add_scalar('Extra/PN-Distribution', train_n/train_tot, 0)
    writer.add_scalar('Extra/PN-Distribution', val_n/val_tot, 1)

    train(model, bcell, optimizer, writer, cfg.train.epochs, cfg.train.device, train_loader, val_loader, tfns, cfg.output.model.dir, run_name)

if __name__ == '__main__':
    cfg = OmegaConf.load('config/train.yaml')
    main(cfg)