# loaddataset.py
# from torchvision.datasets import CIFAR10
from PIL import Image
from torch.utils.data import Dataset
import torch
import glob
import os
import sys
import random

# class PreDataset(Dataset):
#     def __getitem__(self, item):
#         img,target=self.data[item],self.targets[item]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             imgL = self.transform(img)
#             imgR = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return imgL, imgR, target
class TrainDataset(Dataset):
    def __init__(self, root='../data/unlabeled', transform=None, train=True):
        self.transform = transform
        self.train = train
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg"), recursive=True))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.train == True:
            img = Image.open(self.paths[idx])
            # img = Image.fromarray(img)
            if self.transform is not None:
                imgL = self.transform(img)
                imgR = self.transform(img)


            return imgL, imgR
        else:
            img = Image.open(self.paths[idx])
            # img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            return img

class Train2Dataset(Dataset):
    def __init__(self, root='../data/test', transform=None, train=True):
        self.transform = transform
        self.train = train
        self.paths = glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.train == True:
            img = Image.open(self.paths[idx])
            # img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
                label = int(self.paths[idx].split('/')[-2])
                return img, label


class Test(Dataset):
    def __init__(self, root='../data/test', transform=None):
        self.transform = transform
        self.paths = glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            label = int(self.paths[idx].split('/')[-2])
        return img, label

if __name__=="__main__":
    import config
    train_data = TrainDataset(train=True, transform=config.train_transform)
    print((train_data[0][0].shape))

    # root='../data/test'
    # paths = sorted(glob.glob(os.path.join(root, "**/*.jpg"), recursive=True))
    # print(paths[0:10])
    test_data = TestDataset(transform=config.test_transform)
    print((test_data[0][0].shape))