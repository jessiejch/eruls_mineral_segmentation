import torch
# 以下两条语句保证项目根目录下的nets下包的导入
import sys
sys.path.append('E:\\unet-pytorch')
from torchviz import make_dot
import sys 
sys.path.append("..") 
from nets.vgg import *
from nets.stn import *
from nets.resnet import *
from torchvision.models import vgg16

x = torch.randn(1, 3, 512, 512)  # 随机生成一个张量
model = VGG16(pretrained=False)  # 实例化 vgg16，网络可以改成自己的网络
out = model(x)   # 将 x 输入网络
# print(out.shape)
g = make_dot(out)  # 实例化 make_dot
g.view()  # 直接在当前路径下保存 pdf 并打开
# g.render(filename='netStructure/myNetModel', view=False, format='pdf')  # 保存 pdf 到指定路径不打开
