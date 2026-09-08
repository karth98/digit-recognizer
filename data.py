import math
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_img_tensor(df):
    t = torch.tensor(df.values, dtype=torch.float32) / 256.0
    t = t.view(-1, 28, 28)
    t = t.unsqueeze(1)
    t = t.to(device)
    return t

def to_tensor(ser):
    t = torch.tensor(ser.values, dtype=torch.long)
    t = t.to(device)
    return t

def load_data(test_only=False):
    if test_only:
        test_X = pd.read_csv('./input/test.csv')
        test_X = to_img_tensor(test_X)
        return test_X

    train_df = pd.read_csv('./input/train.csv')
    test_X = pd.read_csv('./input/test.csv')

    train_X, train_y = train_df.iloc[:, 1:], train_df.iloc[:, 0]

    train_X, test_X = to_img_tensor(train_X), to_img_tensor(test_X)
    train_y = to_tensor(train_y)

    return train_X, train_y, test_X

def split(X, y):
    X_ = X.cpu()
    y_ = y.cpu()

    X_train, X_test, y_train, y_test = train_test_split(
        X_,
        y_,
        test_size=0.2,
        stratify=y_,
        random_state=42
    )

    X_train = X_train.to(device)
    X_test  = X_test.to(device)
    y_train = y_train.to(device)
    y_test  = y_test.to(device)
    return X_train, y_train, X_test, y_test

def random_mask(img_array, x=5):
    colored_mask = (img_array > 0.2) & (img_array < 1.0)
    rand = torch.rand_like(img_array)
    final_mask = colored_mask & (rand < (x / 100.0))
    img_array[final_mask] = 0.0
    return img_array

def add_gaussian_noise(img, mean=0.1, std=0.2):
    noise = torch.randn_like(img) * std + mean
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0.0, 1.0)

def random_affine_matrix(N, max_rot_deg=20, 
        scale_range=(0.9, 1.1), 
        translate_frac=(0.1, 0.1)):
    # Rotation
    angles = (torch.rand(N, device=device) * 2 - 1) * max_rot_deg
    angles = angles * math.pi / 180.0
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Scale
    scale_x = torch.rand(N, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    scale_y = torch.rand(N, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # Translation (normalized coords)
    tx = (torch.rand(N, device=device) * 2 - 1) * translate_frac[0]
    ty = (torch.rand(N, device=device) * 2 - 1) * translate_frac[1]

    # Affine matrices (N, 2, 3)
    theta = torch.zeros(N, 2, 3, device=device)
    theta[:, 0, 0] = scale_x * cos
    theta[:, 0, 1] = -scale_y * sin
    theta[:, 1, 0] = scale_x * sin
    theta[:, 1, 1] = scale_y * cos
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    return theta

def augment_(X):
    theta = random_affine_matrix(
        X.shape[0],
        max_rot_deg=45,
        scale_range=(0.9, 1.1),
        translate_frac=(0.1, 0.1),
    )
    grid = F.affine_grid(theta, size=X.size(), align_corners=False)
    X_aug = F.grid_sample(
        X,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    X_aug = random_mask(X_aug, x=3)
    X_aug = add_gaussian_noise(X_aug)
    return X_aug

def augment(X, y, k):
    augmented_tensors = [X]  # include the original
    labels_tensors = [y]
    for _ in range(k):
        augmented_tensors.append(augment_(X.clone()))
        labels_tensors.append(y.clone())

    X_aug = torch.cat(augmented_tensors, dim=0)
    y_aug = torch.cat(labels_tensors, dim=0)
    return X_aug, y_aug

def create_loader(train_X, train_y, val_X, val_y):
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1000, shuffle=False)
    return train_loader, val_loader

def create_loader_test(test_X):
    test_dataset = TensorDataset(test_X)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return test_loader

class DigitDataset(Dataset):
    def __init__(self, df, train=True):
        self.train = train
        if train:
            self.labels = df['label'].values
            self.images = df.drop('label', axis=1).values
        else:
            self.labels = None
            self.images = df.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).reshape(1, 28, 28)
        if self.train:
            label = self.labels[idx]
            return img_tensor, label
        else:
            return img_tensor

# def create_loader(df, batch_size=128, train=True):
#     dataset = DigitDataset(df, train=train)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
#     return loader
