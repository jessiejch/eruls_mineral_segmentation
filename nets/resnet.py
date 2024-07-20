import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# 无stn的resnet结构
# Conv2d - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
# dilation=1是正常卷积，dilation=2是空洞卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.LeakyReLU(negative_slope=0.1)
        # self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 在底部创建该类对象时代码如下：
# model = ResNet(Bottleneck, Net, [3, 4, 6, 3], **kwargs)
# 具体结构参看resnet50结构图
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        print("resnet without stn")
        
        #-----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        #-----------------------------------------------------------#
        # 输入图像为RGB,channel为3，定义了第一次conv2d之后的channel为64？
        self.inplanes = 64
        super(ResNet, self).__init__()
        # stage0 开始
        self.conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # batchnorm2d用于解决多层神经网络中间层的协方差偏移问题，类似于网络输入进行零均值化和方差归一化的操作，
        # 64为设定的通道数
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.LeakyReLU(negative_slope=0.1)
        # self.relu   = nn.ReLU()
        # 最大池化，size减半，channel不变
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        
        # size不变, channel加倍，layer[0]为3,block是Bottleneck，下同
        # layer1即stage1
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1, no_of_layer = "layer1")
        # print("layer1:")
        # print(self.layer1)
        # print("----------------------------------------------------")
        # size减半，channel加倍, layer[1]为4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, no_of_layer = "layer2")
        # print("layer2:")
        # print(self.layer2)
        # print("----------------------------------------------------")
        # size减半，channel加倍, layer[2]为6
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, no_of_layer = "layer3")
        # print("layer3:")
        # print(self.layer3)
        # print("----------------------------------------------------")
        # size减半，channel加倍, layer[3]为3
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, no_of_layer = "layer4")
        # print("layer4:")
        # print(self.layer4)
        # print("----------------------------------------------------")

        # Applies a 2D average pooling over an input signal composed of several input planes.
        # kernel_size=7 ,输出channel不变，size减小
        self.avgpool = nn.AvgPool2d(7)
        # full connection layer
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 迭代获取网络中所有的module，第一个是整体的Sequential
        # 网络权重的初始化
        for m in self.modules():
            # 如果是conv2d module,则
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # batch normalization层权重的初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # block为Bottleneck，planes为通道数（四次调用时分别为64,128,256,512), 
    # blocks是数值（四次调用时分别为3,4,6,3）
    def _make_layer(self, block, planes, blocks, stride=1, no_of_layer=0):
        downsample = None
        # 需要进行downsample的情况， bottleneck(block).expansion = 4
        # print("no_of_layer: " + str(no_of_layer))
        # print("planes: "  + str(planes))
        # print("blocks: "  + str(blocks))
        # print("strides: "  + str(stride))
        # print("self.inplanes: "  + str(self.inplanes))
        # print("block.expansion: "  + str(block.expansion))
        if stride != 1 or self.inplanes != planes * block.expansion:
            # print("设置 downsaple！")
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                       kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        # print("----------------------------------------------")
        
        layers = []
        # layer1：block(64,64,1,none)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # 变量前单星号表示将参数转化成元组
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # print(x.size())
        # print("^^^^^^^^^&&&&&&&&************@@@@@@@@@@@@##############$$$$$$$$$$$$")
        
        x       = self.conv1(x)
        x       = self.bn1(x)
        feat1   = self.relu(x)

        x       = self.maxpool(feat1)
        feat2   = self.layer1(x)
        feat3   = self.layer2(feat2)
        feat4   = self.layer3(feat3)
        feat5   = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]
    
# 带两个*的kwargs，形参，表示接受字典类型的参数，变量前双星号表示将参数转化成字典
# 具体值在被调用时确定
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'), strict=False)
    
    del model.avgpool
    del model.fc
    return model


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         # batch normalization使一批feature map满足均值为0，方差为1的分布
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.LeakyReLU(negative_slope=0.1)
#         # self.relu = nn.ReLU()
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out