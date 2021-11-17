from torch.utils.data import Dataset

class PowershellCharDataset(Dataset):
    def __init__(self, char_datas, labels):
        self.char_datas = char_datas
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.char_datas[idx], self.labels[idx]