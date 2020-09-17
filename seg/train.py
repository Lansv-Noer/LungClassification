# public
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations import (CropNonEmptyMaskIfExists, Rotate, Compose, Resize, Normalize)

# private
from models import UNet
from models import DiceLoss
from datasets import ThumbnailDataset, ThumbnailPatchDataset
from tools import onehot, tensor2img


mean_data = [203.81873346 / 255, 116.51206044 / 255, 152.45270792 / 255]
std_data = [38.29048771 / 255, 52.78056623 / 255, 44.04232756 / 255]

aug_train = Compose([
    CropNonEmptyMaskIfExists(496, 496, p=1),
    Rotate((-90, 90), cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101, p=1),
    Normalize(mean=mean_data, std=std_data, p=1),
    ])
aug_val = Compose([
    Normalize(mean=mean_data, std=std_data, p=1),
    ])


def train(path_train: str, path_val: str, epochs: int, batch_size: int, lr: float=0.001, optimizer: str="sgd", save_path: str="."):
    # data set and loader
    dataset_train = ThumbnailDataset(path_train, transform=aug_train)
    dataset_val = ThumbnailDataset(path_val, transform=aug_val)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=True)

    # device and model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(n_channels=3, n_classes=3, bilinear=True).to(device)

    # criteria and loss
    if optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("{} hasn't be implemented!".format(optimizer))

    criteria = DiceLoss()

    # training step
    for epoch in tqdm(range(epochs)):
        model.train()
        loss_epoch = 0
        for idx_batch, (imgs, masks) in enumerate(dataloader_train):
            imgs = imgs.to(device)
            masks = masks.to(device)
            result = model(imgs)
            loss = criteria(result, onehot(masks, dim=1, n_classes=3))
            loss.backward()
            optimizer.step()
            print("Train: loss: {} ep.{}/{} bat.{}/{}".format(loss, epoch+1, epochs, idx_batch+1, len(dataloader_train)))
        torch.save(model.state_dict(), os.path.join(save_path, "temp.pth"))
        model.eval()
        loss_epoch = 0
        for idx_batch, (imgs, masks) in enumerate(dataloader_val):
            imgs = imgs.to(device)
            masks = masks.to(device)
            result = model(imgs)
            loss = criteria(result, onehot(masks, dim=1, n_classes=3))
            print("Val: loss: {} ep.{} bat.{}".format(loss, epoch+1, idx_batch+1))


def evaluate(path_model: str, path_data: str, device: str):
    assert os.path.exists(path_model), "PathError: {} doesn't exist.".format(path_model)
    assert os.path.exists(path_data), "PathError: {} doesn't exist.".format(path_data)

    # augmentation
    aug_val = Compose([
        Normalize(mean=mean_data, std=std_data, p=1),
        ])

    # trained model
    model = UNet(n_channels=3, n_classes=3, bilinear=True)
    state_dict = torch.load(path_model)
    model.load_state_dict(state_dict)
    if device == "cpu":
        device = torch.device("cpu")
        print("Mode: CPU")
    elif device == "cuda" or device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Mode: GPU")
        else:
            device = torch.device("cpu")
            print("Mode: CPU")
            print("gpu is not supported on your computer, so it is switched to cpu mode automaticly")
    else:
        raise NameError("device must be \"cpu\",\"gpu\",\"cuda\"")
    model = model.to(device)

    # dataset
    dataset = ThumbnailPatchDataset(dir=path_data, transform=aug_val, sizeTile=[496, 496])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    model.eval()
    with torch.no_grad():
        num_loss = 0
        loss_all = torch.tensor([0], dtype=torch.float64)

        for image, mask, (name, x, y) in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            result = model(image)


if __name__ == '__main__':
    # training
    # path_train = "G:\\medical\\Out\\Train"
    # path_val = "G:\\medical\\Out\\VAl"
    # train(path_train, path_val, epochs=50, batch_size=4, lr=0.001, optimizer="sgd")

    # evaluating
    path_load = "./temp.pth"
    evaluate(path_load, path_data="G:\\medical\\Out\\Val", device="cuda")
    print("End.")
