import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

# 本文件作用：
# 1. 划分测试和验证集，存放在txt文件中，随机选取
# 2. 检查数据集格式是否正确，image为jpg，mask为png
# 3. mask中标记区域的值为对应的类别？


#-------------------------------------------------------#
#   trainval_percent：训练集验证集的数量比例，这里是全部 
#   train_percent：训练集数量比例
#   
#   将准备的数据集划分为训练集training set和验证集val set
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.7
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
# VOCdevkit_path      = 'VOCdevkit_noaugment'
VOCdevkit_path      = 'VOCdevkit'

# 生成测试集和验证集的txt文件
if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    # training set中label存放地址，因为image和label同名，所以也代表了所有image
    # segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    print(VOCdevkit_path)
    segfilepath     = os.path.join(VOCdevkit_path, 'SegmentationClass')
    # 生成的train和val划分的txt存放位置
    # saveBasePath    = os.path.join(VOCdevkit_path, '/VOC2007/ImageSets/Segmentation')
    saveBasePath    = os.path.join(VOCdevkit_path, 'ImageSets/Segmentation')
    
    # 所有label文件列表
    temp_seg = os.listdir(segfilepath)
    # 加入后缀名后的label文件列表
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)                                   # mask（png）文件的数量
    all_list    = range(num)                                   # range(0,num),与出现在for中同义，得到一个list，[0,1,2...num-1]
    num_trainval   = int(num*trainval_percent)                 # 训练集验证集的数量，这里是全部
    num_train      = int(num_trainval*train_percent)           # 训练验证集的总数乘以训练集比例，得到训练集数量
    trainval= random.sample(all_list,num_trainval)             # 从全局数据集all_list中随机取样，数量为tv(训练验证集数量)， 这里是全部取，得到list
    train   = random.sample(trainval,num_train)                # 再从trainval随机取样, 数量为tr，得到训练集序列
    
    print("train and val size",num_trainval)
    print("train size",num_train)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    # 将之前生成的序列，读取对应的文件名称，写入txt文件中
    for i in all_list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

# 检查数据集格式
    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums        = np.zeros([256], np.int32)
    for i in tqdm(all_list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        # mask,也就是标签图片应为8位宽彩图或灰度图,维度不超过2,否则非法
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        # 将每一张mask重整为长度为256的一维array，统计每个值（代表类别）的数量，所有array再累加
        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
     
    # 打印在数据集(mask)中所有类别像素点的数量，类别用0,1,2,3……表示，与json_to_dataset.py中的classes顺序对应
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")