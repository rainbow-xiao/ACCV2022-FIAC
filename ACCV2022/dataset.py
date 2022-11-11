import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset

class ImageEmbedding_Dataset(Dataset):
    def __init__(self, df, split, mode, img_size):
        self.mode = mode
        if self.mode == 'train':
            self.df = df[df['fold'] != split].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['fold'] == split].reset_index(drop=True)
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_files'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = self.transforms(image=image)['image']
        images = self.norm(images)
        images = torch.from_numpy(images.transpose((2,0,1)))
        label = row['labels']
        label = torch.as_tensor(label)
        return images, label
    
    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
        
    def get_transforms(self,):
        if self.mode == 'train':
            transforms=(A.Compose([
#                 A.Resize(570, 570),
                A.RandomResizedCrop (self.img_size, self.img_size, scale=(0.8, 1.0), ratio=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0., scale_limit=[0.0, 0.3], rotate_limit=10, border_mode=0, p=0.6),
                A.CoarseDropout(max_holes=1, max_height=int(self.img_size * 0.2),max_width=int(self.img_size * 0.2), 
                                min_holes=1, min_height=int(self.img_size * 0.1),min_width=int(self.img_size * 0.1), 
                                fill_value=0, p=0.5),
            ]))
        else:
            transforms=(A.Compose([A.Resize(int(self.img_size*1.5625), int(self.img_size*1.5625)),
                               A.CenterCrop(self.img_size, self.img_size)]))
        return transforms
    
class ACCV_test(Dataset):
    def __init__(self, df, img_size):
        self.df = df
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['id'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = self.transforms(image=image)['image']
        images = self.norm(images)
        images = torch.from_numpy(images.transpose((2,0,1)))
        return images
    
    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
        
    def get_transforms(self,):
        transforms=(A.Compose([A.Resize(self.img_size, self.img_size),
        return transforms

