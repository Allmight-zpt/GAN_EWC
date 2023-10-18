import os

import numpy as np
import torch
from torchvision.transforms import transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader


# 从数据集中随机采样
def sample_batch_index(total, batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


# 获取Mnist数据集
def getMnist(dataRoot, batchSize, getDataSet=False):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    Mnist = dset.MNIST(dataRoot, transform=img_transform, download=True)
    Mnist_dataloader = DataLoader(Mnist, batch_size=batchSize, shuffle=True, num_workers=0)
    if getDataSet:
        return Mnist_dataloader, Mnist
    return Mnist_dataloader


# 获取Fashion Mnist数据集
def getFashionMnist(dataRoot, batchSize, getDataSet=False):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    FashionMnist = dset.FashionMNIST(dataRoot, transform=img_transform, download=True)
    FashionMnist_dataloader = DataLoader(FashionMnist, batch_size=batchSize, shuffle=True, num_workers=0)
    if getDataSet:
        return FashionMnist_dataloader, FashionMnist
    return FashionMnist_dataloader


# 参数初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 构造工作目录
def createWorkDir(workDirName):
    if not os.path.exists(workDirName):
        os.makedirs(workDirName)

    if not os.path.exists(workDirName + '/models'):
        os.makedirs(workDirName + '/models')

    if not os.path.exists(workDirName + '/gen_images'):
        os.makedirs(workDirName + '/gen_images')


# 数据预处理
def preProcess(dataloader):
    images = []
    labels = []
    for i, data in enumerate(dataloader):
        image, label = data
        images.append(image)
        labels.append(label)
    images = torch.cat(images)
    print(images.shape)
    return images


# 将一个batch的图片转换为一张图片
def batchToOne(batchImage, row, col):
    res = torch.tensor([])
    for i in range(row):
        temp_res = torch.tensor([])
        for j in range(col):
            temp_res = torch.cat((temp_res, batchImage[i * col + j]), dim=2)
        res = torch.cat((res, temp_res), dim=1)
    return res
