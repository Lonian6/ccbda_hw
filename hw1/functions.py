import os
import numpy as np
from PIL import Image, ImageOps
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from random import sample, choices, randint
# from typing import Tuple
# from vidaug import augmentors as va

IMG_SIZE = 224
FRAME_NUMBER = 29
CLASS_NUM = 39
## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

## ---------------------- Dataloaders ---------------------- ##
class dataset(data.Dataset):
    # data loading
    def __init__(self, video_list, labels_list, transform=None):
        self.video_list = video_list
        self.labels_list = labels_list
        self.transform = transform

    # working for indexing
    def __getitem__(self, index):
        video_path = self.video_list[index]
        label = torch.LongTensor([self.labels_list[index]])
        # 把video取FRAME_NUMBER的幀數
        frames = self.load_video(video_path)    # tensor size = (FRAME_NUMBER, 3, IMG_SIZE, IMG_SIZE)
        return frames, label


    # return the length of our dataset
    def __len__(self):
        return len(self.video_list)

    # 算影片frame數量
    def count_frames(self, path):
        vcap = cv2.VideoCapture(path)
        if vcap.isOpened():
            return(int(vcap.get(7)))
        else:
            return 0
    
    def load_video(self, path, resize=(IMG_SIZE, IMG_SIZE)):
        cap = cv2.VideoCapture(path)
        frame_num = self.count_frames(path)
        num = list(range(frame_num))
        if frame_num < FRAME_NUMBER:
            num += choices(num, k = FRAME_NUMBER-frame_num)
        else:
            num = sample(num, FRAME_NUMBER)
        num.sort()

        frames = []
        count_num = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # frame = self.crop_center_square(frame)
                frame = self.crop_random_square(frame)
                frame = cv2.resize(frame, resize)
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # frame = frame[:, :, [2, 1, 0]]
                frame = self.transform(frame)

                append_number = num.count(count_num)
                for i in range(append_number):
                    frames.append(frame)
                count_num += 1
        finally:
            cap.release()
        return torch.stack(frames, dim=0)
    
    def crop_random_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = randint(0, x-min_dim)
        start_y = randint(0, y-min_dim)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

class aug_dataset(data.Dataset):
    # data loading
    def __init__(self, video_list, labels_list, transform):
        self.video_list = video_list
        self.labels_list = labels_list
        self.transform = transform

    # def _load_videolist(self):
    #     self.loaded_data = []
    #     video_path = self.video_list[index]
    #     label = torch.LongTensor([self.labels_list[index]])
    #     # 把video取FRAME_NUMBER的幀數
    #     frames = self.load_video(video_path)    # tensor size = (FRAME_NUMBER, 3, IMG_SIZE, IMG_SIZE)
    #     return frames, label

    # working for indexing
    def __getitem__(self, index):
        video_path = self.video_list[index//len(self.transform)]
        t = self.transform[index % len(self.transform)]
        label = torch.LongTensor([self.labels_list[index//len(self.transform)]])
        # 把video取FRAME_NUMBER的幀數
        frames = self.load_video(video_path, t)    # tensor size = (FRAME_NUMBER, 3, IMG_SIZE, IMG_SIZE)
        return frames, label


    # return the length of our dataset
    def __len__(self):
        return len(self.transform)*len(self.video_list)

    # def padding(img, expected_size):
    #     desired_size = expected_size
    #     delta_width = desired_size - img.size[0]
    #     delta_height = desired_size - img.size[1]
    #     pad_width = delta_width // 2
    #     pad_height = delta_height // 2
    #     padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    #     return ImageOps.expand(img, padding)

    def resize_with_padding(self, img, expected_size):
        img.thumbnail((expected_size[0], expected_size[1]))
        # print(img.size)
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return ImageOps.expand(img, padding)

    # 算影片frame數量
    def count_frames(self, path):
        vcap = cv2.VideoCapture(path)
        if vcap.isOpened():
            return(int(vcap.get(7)))
        else:
            return 0
    
    def load_video(self, path, tsf):
        cap = cv2.VideoCapture(path)
        frame_num = self.count_frames(path)
        num = list(range(frame_num))
        if frame_num < FRAME_NUMBER:
            num += choices(num, k = FRAME_NUMBER-frame_num)
        else:
            num = sample(num, FRAME_NUMBER)
        num.sort()

        frames = []
        count_num = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # frame = self.crop_center_square(frame)
                # frame = self.crop_random_square(frame)
                # frame = cv2.resize(frame, resize)
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.resize_with_padding(frame, (122, 122))
                
                # frame = frame[:, :, [2, 1, 0]]
                frame = tsf(frame)

                append_number = num.count(count_num)
                for i in range(append_number):
                    frames.append(frame)
                count_num += 1
        finally:
            cap.release()
        return torch.stack(frames, dim=0)

## ---------------------- end of Dataloaders ---------------------- ##


## -------------------- (reload) model prediction ---------------------- ##
def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred
## -------------------- end of model prediction ---------------------- ##

## ------------------------ CRNN module ---------------------- ##
def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

class EncoderCNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.5, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),                      
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq
class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.2, num_classes=CLASS_NUM):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x
## ---------------------- end of CRNN module ---------------------- ##
