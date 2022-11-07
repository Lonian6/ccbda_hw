# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18
from torchsummary import summary
import numpy as np
EMBEDDING_SIZE = 512
# stage one ,unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRStage1, self).__init__()

        self.f = []
        # for name, module in resnet18().named_children():
            # if name == 'conv1':
            #     module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            #     self.f.append(module)
        self.f.append(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0))
        self.f.append(nn.ReLU(inplace=True))
        # self.f.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.f.append(nn.Conv2d(64, 128, kernel_size=3, stride=2))
        self.f.append(nn.ReLU(inplace=True))
        # self.f.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.f.append(nn.Conv2d(128, 256, kernel_size=3, stride=2))
        self.f.append(nn.ReLU(inplace=True))
        self.f.append(nn.Conv2d(256, 256, kernel_size=3, stride=2))
        self.f.append(nn.ReLU(inplace=True))
        # self.f.append(nn.Conv2d(128, 128, kernel_size=3, stride=2))
        # self.f.append(nn.ReLU(inplace=True))
        # self.f.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.f.append(nn.Flatten())
        self.f.append(nn.Linear(6400, 512))
        self.f.append(nn.ReLU(inplace=True))
        # self.f.append(nn.Linear(1024, 512))
        # self.f.append(nn.ReLU(inplace=True))
        # self.f.append(nn.Linear(EMBEDDING_SIZE))
        # self.f.append(nn.ReLU(inplace=True))
        # encoder
        # self.f.append(module)
        self.f = nn.Sequential(*self.f)
        # projection head
        # self.g = nn.Sequential(nn.Linear(2048, EMBEDDING_SIZE, bias=False),
        #                        nn.BatchNorm1d(EMBEDDING_SIZE),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(EMBEDDING_SIZE, feature_dim, bias=True))
        self.g = nn.Sequential(nn.Linear(512, 512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 512),
                               nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# stage two ,supervised learning
class SimCLRStage2(torch.nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)
        # self.train = train

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return feature, out


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,out_1,out_2,batch_size,temperature=0.5):
        N, C = out_1.shape

        z = torch.cat([out_1, out_2], dim=0)           # [2N, C]
        z = F.normalize(z, p=2, dim=1)                 # [2N, C]
        s = torch.matmul(z, z.t()) / temperature       # [2N, 2N] similarity matrix
        mask = torch.eye(2 * N).bool().to(z.device)    # [2N, 2N] identity matrix
        s = torch.masked_fill(s, mask, -float('inf'))  # fill the diagonal with negative infinity
        label = torch.cat([                            # [2N]
            torch.arange(N, 2 * N),                    # {N, ..., 2N - 1}
            torch.arange(N),                           # {0, ..., N - 1}
        ]).to(z.device)

        loss = F.cross_entropy(s, label)               # NT-Xent loss
        return loss
        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        # [2*B, D]

        # out = torch.cat([out_1, out_2], dim=0)
        # # [2*B, 2*B]
        # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # # [2*B, 2*B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # # 分子： *为对应位置相乘，也是点积
        # # compute loss
        # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # # [2*B]
        # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        # return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # inputs.to(device)
    # for name, module in resnet18().named_children():
    #     print(name, module)
    net = SimCLRStage1().to(device)
    # for name,parameters in net.named_parameters():
    #     print(name,':',parameters.size())
    summary(net, (3, 96, 96))