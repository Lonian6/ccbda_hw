# test.py
import torch,argparse
import net,config
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import net, config, loaddataset
WORKER_NUM = 8

def test(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    transform = config.test_transform

    model = net.SimCLRStage2(num_class=4).to(DEVICE)
    model.load_state_dict(torch.load(args.pre_model, map_location='cpu'), strict=False)

    # unlabel embedding
    if args.unlabel_f == "True":
        unlabel_emb = []
        train_dataset=loaddataset.TrainDataset(train=False, transform=config.test_transform)
        train_data=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=False, num_workers=WORKER_NUM)
        for batch,img in enumerate(train_data):
            img = img.to(DEVICE)
            embedding, _ = model(img)
            for i in range(embedding.shape[0]):
                unlabel_emb.append(embedding[i].to("cpu").numpy())
        unlabel_emb = np.array(unlabel_emb)
        np.save("0717072.npy", unlabel_emb)
        print(unlabel_emb.dtype)
        print(unlabel_emb.shape)
    else:
        print("Unlabeled feature skip")

    # test embedding
    if args.test_f == "True":
        test_emb = []
        test_label = []

        test_dataset=loaddataset.Test(transform=config.test_transform)
        test_data=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False, num_workers=WORKER_NUM)

        for batch,(img,label) in enumerate(test_data):
            img = img.to(DEVICE)
            embedding, _ = model(img)
            for i in range(embedding.shape[0]):
                test_emb.append(embedding[i].to("cpu").numpy())
                test_label.append(label[i])
        # test1_emb = np.array(test_emb)
        # print("test1_emb :",test1_emb.shape)

        test_emb = torch.Tensor(test_emb)
        test_label = torch.FloatTensor(test_label)
        print("test_emb :",test_emb.shape)
        print("test_label :",test_label.shape)

        acc = KNN(test_emb, test_label, batch_size=16)
        print("Accuracy: %.5f" % acc)
    else:
        print("KNN test skip")
        
    

def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test SimCLR')
    parser.add_argument('--pre_model', default="./pth/model_stage2_epoch150.pth", type=str, help='')
    parser.add_argument('--unlabel_f', default="True", type=str, help='')
    parser.add_argument('--test_f', default="True", type=str, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='')

    args = parser.parse_args()
    # embedding = np.load('0717072.npy', allow_pickle=True)
    # print(embedding.shape)
    # print(embedding[0].shape[0])
    test(args)