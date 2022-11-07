# config.py
import os
from torchvision import transforms

use_gpu=True
gpu_name=0

pre_model=os.path.join('pth','model.pth')

save_path="pth"

train1_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomRotation(90, expand=True),
    transforms.CenterCrop(96),
    transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),
    transforms.ToTensor()])#,
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor()])