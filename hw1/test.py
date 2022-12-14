import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import glob
import sys

# set path
# data_path = "./jpegs_256/"    # define UCF-101 RGB data path
# action_name_path = "./UCF101actions.pkl"
save_model_path = "./CRNN_ckpt/"

# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
img_x, img_y = 256, 342  # resize video 2d frame size
dropout_p = 0.2       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 39             # number of target category
batch_size = 50
# Select which frame to begin & end in videos
# begin_frame, end_frame, skip_frame = 1, 29, 1


# load UCF101 actions names
action_names = list(range(39))

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # # example
# # y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# # y_onehot = labels2onehot(enc, le, y)
# # y2 = onehot2labels(le, y_onehot)

# actions = []
# fnames = os.listdir(data_path)

# all_names = []
# for f in fnames:
#     loc1 = f.find('v_')
#     loc2 = f.find('_g')
#     actions.append(f[(loc1 + 2): loc2])

#     all_names.append(f)
all_names = glob.glob(r'/ccbda/hw1/data/test/*.mp4')
fnames = os.listdir('/ccbda/hw1/data/test/')

actions = [0]*len(all_names)
# list all data files
all_X_list = all_names              # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

# reset data loader
all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
all_data_loader = data.DataLoader(dataset(all_X_list, all_y_list, transform=transform), **all_data_params)


# reload CRNN model
cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(sys.argv[1]))))
rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(sys.argv[1]))))
print('CRNN model reloaded!')


# make all video predictions by reloaded model
print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
all_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)


# write in pandas dataframe
df = pd.DataFrame(data={'name': fnames, 'label': cat2labels(le, all_y_pred)})
df.to_csv("./result.csv", index=False)  # save pandas dataframe
# pd.read_pickle("./all_videos_prediction.pkl")
print('video prediction finished!')