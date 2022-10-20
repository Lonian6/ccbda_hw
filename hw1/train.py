# from collections import _T1
import time
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from random import sample
from random import choices
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import sys
# from vidaug import augmentors as va

# set path
# data_path = "./jpegs_256/"    # define UCF-101 RGB data path
# action_name_path = './UCF101actions.pkl'
save_model_path = "./CRNN_ckpt/"
# print(os.path.join(save_model_path, 'CRNN_epoch_test_score.npy'))
if os.path.isdir(save_model_path):
    print("folder exist !")
else:
    os.mkdir(save_model_path)
    print("folder created !")
# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512      # latent dim extracted by 2D CNN
img_x, img_y = 256, 342  # resize video 2d frame size
dropout_p = 0.5          # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 39              # number of target category
epochs = 100        # training epochs
if len(sys.argv) < 2:
    continue_state = False
    start_epoch = 0
else:
    continue_state = True
    start_epoch = int(sys.argv[1])

print(continue_state, start_epoch)

batch_size = 32
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
# begin_frame, end_frame, skip_frame = 1, 29, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        start = time.time()
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()
        
        end = time.time()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%, Time: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score, end - start))
    print('train done')
    return losses, scores


def validation(model, device, optimizer, test_loader):
    start = time.time()
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))
    end = time.time()
    print('Time: {}'.format(end-start))
    return test_loss, test_score


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# load UCF101 actions names
action_names = list(range(CLASS_NUM))

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)
all_names = glob.glob(r'/ccbda/hw1/data/train/*/*.mp4')
# print(all_names[0])
actions = []
for i in all_names:
    actions.append(int(i.split('/')[-2]))


# list all data files
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size = 0.2, random_state = 42, stratify = all_y_list)

transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
t1 = transforms.Compose([   transforms.RandomGrayscale(0.1), 
                            transforms.RandomPerspective(0.4,0.6),
                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                            transform])

# t0=transform
# t1=transforms.Compose([t0,transforms.RandomPerspective(distortion_scale=0.4, p=1.0)])
# t2=transforms.Compose([transforms.Resize((128,128)),transforms.RandomCrop(96),t0])
# t3=transforms.Compose([t0,transforms.RandomRotation((0,180))])
# t4=transforms.Compose([t2,transforms.RandomRotation((0,180))])
# t5=transforms.Compose([t2,transforms.RandomPerspective(distortion_scale=0.4, p=1.0)])
# t6=transforms.Compose([t2,transforms.RandomPerspective(distortion_scale=0.4, p=1.0),transforms.RandomPerspective(distortion_scale=0.4, p=1.0)])

transform_tr = [transform, t1, t1]
# selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
# sometimes = lambda aug: va.Sometimes(0.5, aug)
# seq = va.Sequential([
#     va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
#     sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
# ])

# train_set = dataset(train_list, train_label, transform=transform)
# print('length of training dataset: ', len(train_set))
train_set = aug_dataset(train_list, train_label, transform=transform_tr)
print('length of training augmentation dataset: ', len(train_set))
valid_set = dataset(test_list, test_label, transform=transform)
# os._exit()


train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


# Create model
cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                        drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                        h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

if continue_state == True:
    cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(start_epoch))))
    rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(start_epoch))))

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
# optimizer = torch.optim.Adam(crnn_params, lr = learning_rate, weight_decay=0.001)

optimizer = torch.optim.SGD(crnn_params, lr = learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], gamma=0.1)

if continue_state == True:
    optimizer.load_state_dict(torch.load(os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(start_epoch))))
else:
    start_epoch = 0


# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    epoch = epoch + start_epoch
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)
    
    scheduler.step()

# plot
# fig = plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
# plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
# plt.title("model loss")
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(['train', 'test'], loc="upper left")
# # 2nd figure
# plt.subplot(122)
# plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
# plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
# plt.title("training scores")
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(['train', 'test'], loc="upper left")
# title = "./fig_UCF101_CRNN.png"
# plt.savefig(title, dpi=600)
# # plt.close(fig)
# plt.show()