import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

class FER2013Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_col = 'emotion' if 'emotion' in self.df.columns else 'label'
        self.has_labels = self.label_col in self.df.columns
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ').reshape(48, 48)
        
        if self.has_labels:
            label = int(row[self.label_col])
            target = torch.tensor(label, dtype=torch.long)
        else:
            target = torch.tensor(-1, dtype=torch.long)
            
        if self.transform:
            pixels = self.transform(pixels)
        else:
            pixels = transforms.ToTensor()(pixels)
            
        return pixels, target

def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transform, val_transform

def get_train_val_loaders(csv_path, batch_size=128, val_split=0.2):
    train_transform, val_transform = get_transforms()
    
    train_dataset = FER2013Dataset(csv_path, transform=train_transform)
    val_dataset = FER2013Dataset(csv_path, transform=val_transform)
    
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader