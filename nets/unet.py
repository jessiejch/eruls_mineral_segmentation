import torch
import torch.nn as nn

from nets.resnet_stn import resnet50
# from nets.resnet import resnet50
from nets.vgg import VGG16
# from nets.stn import Net

# 解码部分上采样基本操作单元cat->conv1->relu->conv2->relu
# 上采样采用线性插值
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # print()
        # print(self._get_name)
        # print("in_size:" + str(in_size))
        # print("out_size:" + str(out_size))
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        # scale_factor : multiplier for spatial size, 空间缩放因子
        # 该操作只是size（height和width）加倍，通道不发生变化
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        # 若使用转置卷积，使用以下操作得到size加倍的up
        # 相比差值上采样，转置卷积有参数需要学习        
        # self.up     = nn.ConvTranspose2d(upconv_size, upconv_size, kernel_size=3,stride=2,padding=1)
        # print("-----------------------")
        # print(in_size)
        # print(out_size)
        # print(in_size-out_size*2)
        # print("-----------------------")
        self.relu   = nn.LeakyReLU(0.1)
        # self.relu   = nn.ReLU()
    
    # 代表一次合并及上采样，两次卷积
    def forward(self, inputs1, inputs2):
        # 1表示dimension
        # 如果使用UpsamplingBilinear2d，用此行concat
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # 如果是使用nn.ConvTranspose2d,可以这样concat
        # print()
        # print("inputs1:",inputs1.size())
        # print("inputs2:",inputs2.size())
        # print("inputs2 - up:",self.up(inputs2).size())
        # print("ConvTranspose2d(inputs1): ",self.up(inputs2,output_size=inputs1.size()).size())
        # outputs = torch.cat([inputs1, self.up(inputs2,output_size=inputs1.size())], 1)        # 若使用nn.ConvTranspose2d,则RuntimeError
        # print("---------------------------------------------------------")
        # print("concat" + str(inputs1.size()) + " " + str(inputs2.size()))
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        
        # 如果将上采样的主要方法改为转置卷积，需要显式指定每次上采样的通道数
        # 通过程序运行和显式，每次和输入输出匹配的转置卷积通道数如下所示，将作为参数传入unetUp进行初始化
        # upConv_filters = [128,256,512,2048]
        # upsampling
        # 64,64,512
        # 一个上采样操作单元，包括cat->conv1->relu->conv2->relu，
        # 若vgg：unetUP(1024,512)，通道从1024变为512，下同
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        # 一个上采样操作单元，若vgg：unetUP(768,256)
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        # 一个上采样操作单元，若vgg：unetUP(384,128)
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        # 一个上采样操作单元，若vgg：unetUP(192,64)
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                # 线性插值改为转置卷积
                # nn.ConvTranspose2d(out_filters[0], out_filters[0], kernel_size=3,stride=2,padding=1,output_padding=1), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                # nn.ReLU(),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                # nn.ReLU(),
                nn.LeakyReLU(0.1),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    # Unet整体的前向传播
    def forward(self, inputs):
        # print(inputs.size())
        # Encoder部分，U的左半边，得到5个特征图
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        # Decoder部分，U的右半部分，上采样并分别与特征图融合
        # 若为vgg:          channels         size
        #          feat5      512           32*32
        #          feat4      512           64*64
        #          feat3      256           128*128
        #          feat2      128           256*256
        #          feat1      64            512*512
       
        up4 = self.up_concat4(feat4, feat5)        
        up3 = self.up_concat3(feat3, up4)  
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        # print()
        # print("feat5:",feat5.size())
        # print("feat4:",feat4.size())
        # print("feat3:",feat3.size())
        # print("feat2:",feat2.size())
        # print("feat1:",feat1.size())
        # print("up4：",up4.size())
        # print("up3：",up3.size())
        # print("up2：",up2.size())
        # print("up1：",up1.size())
        # print()
        # backbone为vgg即为none的情况，不执行up_conv
        # backbone为resnet即为非none的情况，执行up_conv

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    # 如果冻结backbone，Encoder部分暂不更新
    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    # 如果不冻结backbone，Encoder和Decoder一起更新
    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
