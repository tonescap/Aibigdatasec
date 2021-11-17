import torch
from torch import nn
from torch.nn import functional as F

class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = (62, 3),
                stride=1, padding=0)
        torch.nn.init.normal_(self.conv1.weight)
        self.max1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.fc1 = nn.Sequential(nn.Linear(43520, 1024), nn.ReLU(), nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.5))
        self.fc3 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x)).squeeze()
        x = self.max1(x).flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class TokenCNN(nn.Module):
    def __init__(self, embeddings_pretrained = None):
        super(TokenCNN, self).__init__()
        
        if embeddings_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings_pretrained)
#             self.embedding.weight.requires_grad=False
        else:
            self.embedding = nn.Embedding(5396, 32)

        self.CNN = nn.Sequential(
            nn.Conv1d(32, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(1998, 1),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(128, 1)
        
    def forward(self, input):
        x = self.embedding(input).permute(0,2,1)
        x = self.CNN(x).flatten(start_dim=1)
        return self.fc(x)