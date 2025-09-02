from datasets import load_dataset
import numpy as np
import pandas as pd 
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms


def pixels_to_image(pixels_str):
    pixels = np.array(pixels_str.split(), dtype=np.uint8)
    img = pixels.reshape(48, 48)
    return img 

class FERDataset(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform

    def len(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        pixels_str = self.df.iloc[idx]['pixels']
        img = pixels_to_image(pixels_str).convert("L")
        label = int(self.df.iloc[idx]['emotion'])

        if self.transform:
            img = self.transfrom(img)
        return label, img

def get_dataloader(batch_size = 64, val_ratio = 0.1):
    dataset = load_dataset("Jeneral/fer-2013")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(48, 48),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])
    full_train_dataset = FERDataset(train_df, transform=transform)

    #tách train để có tập validation 
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])


    test_dataset = FERDataset(test_df, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, val_loader, test_loader