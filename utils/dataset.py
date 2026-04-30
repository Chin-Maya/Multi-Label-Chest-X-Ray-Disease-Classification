# utils/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from pathlib import Path

class ChestXRayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, labels=CFG.DISEASE_LABELS):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['Image Index']
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((CFG.IMG_SIZE, CFG.IMG_SIZE))  # fallback resize
        image = A.to_numpy(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(row[self.labels].values.astype(float), dtype=torch.float32)
        
        return image, label


# Default transforms
def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(CFG.IMG_SIZE, CFG.IMG_SIZE, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(),
        ])


        