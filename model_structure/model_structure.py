from torchsummary import summary
import sys
sys.path.append('E:\\unet-pytorch')
from nets.vgg import *
from nets.stn import *
from nets.unet import *
from nets.resnet import *

# myNet = Net([3,6,10],[10,30,30])
# myNet = Net([64,128,256],[256,6,6])
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',512,512,512,'M']
}
# myNet = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = 3))  # 本项目VGG网络结构
# myNet = Unet(num_classes=9, pretrained=False, backbone="vgg")
myNet = ResNet(Bottleneck,[3, 4, 6, 3])
myNet.to('cuda')
# summary(myNet,(1,28,28))
summary(myNet,(3,512,512))  # 输出网络结构
# summary(myNet, (64,128,128))
