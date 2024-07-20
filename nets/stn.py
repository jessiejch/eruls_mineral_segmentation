import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self,channels,fea_size):
        # 如果需要在resnet不同阶段应用stn，考虑将channels，fc1第一个参数，fc_loc第一个linear的维度参数初始化时作为参数传入
        super(Net, self).__init__()
        # self.xs=np.array([1,1,1,1])  
        # torch.from_numpy(self.xs) 
        # Spatial transformer localization-network
        # 因为pytorch tutorial中源代码是针对较小size的minst(28*28)图像，本实验需要针对较大尺寸(512*512)图像，
        # 所以对localization中参数进行了调整。
        self.localization = nn.Sequential(
            # 针对（3,512,512），3为可能需要修改的参数
            # nn.Conv2d(3, 6, kernel_size=7),
            # 针对（64,128,128），建议修改如下：
            nn.Conv2d(channels[0], channels[1], kernel_size=7),

            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),

            # 针对（3,512,512），6和10为可能需要修改的参数
            # nn.Conv2d(6, 10, kernel_size=5),
            # 针对（64,128,128），建议修改如下：
            nn.Conv2d(channels[1], channels[2], kernel_size=5),

            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            # 针对（3,512,512），10和124是可能需要修改的参数
            # nn.Linear(10 * 30 * 30, 32),
            # 针对（64,256,256），建议修改如下：
            # nn.Linear(256*6*6, 32),
            nn.Linear(fea_size[0]*fea_size[1]*fea_size[2], 32),
            # 是否应该改为如下，但需要获取xs的维度
            # nn.Linear(xs.size()[1]*xs.size()[2]*xs.size()[3], 32)
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # self.fc_loc = nn.Sequential()
        # self.fc_loc.add_module(nn.Linear(self.xs.size()[1]*self.xs.size()[2]*self.xs.size()[3], 32))
        # self.fc_loc.add_module(nn.ReLU(True))
        # self.fc_loc.add_module(nn.Linear(32, 3 * 2))



        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # print("self.fc_loc[0]: " + str(self.fc_loc[0]))
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.cuda.FloatTensor([1, 0, 0, 0, 1, 0]))

    # Spatial transformer network
    # x是一张原始图像（特征图） 
    def stn(self, x):
        # print("x初始size：" + str(x.size()))
        xs = self.localization(x)
        # print(type(xs))
        self.xs = xs
        # print(self.xs.size())
        # print("-------------------------------------------------------")
        # print("localization 后的size(xs): " + str(xs.size()))
        # view是重构张量的维度，如果出现-1表示该维由总维度和其他维度计算而来
        # xs = xs.view(-1, 10 * 3 * 3)
        # 此写法是不是更有通用性
        xs = xs.view(-1, xs.size()[1]*xs.size()[2]*xs.size()[3])
        # print("view 后的size(xs): " + str(xs.size()))
        # nn.Sequential(nn.Linear(10 * 3 * 3, 32),nn.ReLU(True),nn.Linear(32, 3 * 2)
        theta = self.fc_loc(xs)
        # print("self.fc_loc 后的size(xs->theta): " + str(theta.size()))
        theta = theta.view(-1, 2, 3)
        # print("view 后的size(theta): " + str(theta.size()))

        # theta (Tensor) – input batch of affine matrices with shape (N×2×3) for 2D or (N×3×4) for 3D
        # size (torch.Size) – the target output image size. (N×C×H×W for 2D or N×C×D×H×W for 3D) Example: torch.Size((32, 3, 24, 24))
        # 最后一个param可选
        grid = F.affine_grid(theta, x.size(), align_corners = True)
        # print("affine_grid 后的size(grid): " + str(grid.size()))
        x = F.grid_sample(x, grid, align_corners = True)
        # print("grid_sample 后的size(x): " + str(x.size()))

        return x
    
    # forward function
    def forward(self, x):
        # transform the input
        x = self.stn(x)
        # print(x.size())
        # Perform the usual forward pass
        # 使用torchsummary.summary查看网络结构时，F.relu, F.max_pool2d, F.dropout均未显示在结构中
        # self.conv1(x),执行对象的forward方法。
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 625000)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x


