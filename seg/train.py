# public
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations import (CropNonEmptyMaskIfExists, Rotate, Compose, Resize, Normalize)

from models import UNet
from datasets import ThumbnailDataset


mean_data = [203.81873346 / 255, 116.51206044 / 255, 152.45270792 / 255]
std_data = [38.29048771 / 255, 52.78056623 / 255, 44.04232756 / 255]

aug_train = Compose([
    CropNonEmptyMaskIfExists(496, 496, p=1),
    Rotate((-90, 90), cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101, p=1),
    Normalize(mean=mean_data, std=std_data, p=1),
    ])
aug_val = Compose([
    CropNonEmptyMaskIfExists(496, 496, p=1),
    Normalize(mean=mean_data, std=std_data, p=1),
    ])


def train(path_train: str, path_val: str, epochs: int, batch_size: int, lr: float=0.001, optimizer: str="sgd"):
    # data set and loader
    dataset_train = ThumbnailDataset(path_train, transform=aug_train)
    dataset_val = ThumbnailDataset(path_val, transform=aug_val)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=True)

    # device and model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(3, 3, True).to(device)

    # criteria and loss
    if optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("{} hasn't be implemented!".format(optimizer))

    criteria = nn.MSELoss()

    # training step
    model.train()
    for epoch in range(epochs):
        loss_epoch = 0
        for idx_batch, (imgs, masks) in tqdm(dataloader_train):
            imgs = imgs.to(device)
            masks = masks.to(device)
            result = model(imgs)
            loss = criteria(result, masks)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    path_train = "I:\\Out\\Train"
    path_val = "I:\\Out\\val"
    train(path_train, path_val, epoch=20, batch_size=4, lr=0.001, optimizer="sgd")
    print("End.")
