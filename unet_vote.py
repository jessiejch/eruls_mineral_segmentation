import copy
from collections import Counter
import numpy as np
from PIL import Image
from unet import Unet
from utils.utils import cvtColor
import colorsys
import os
import math
import pandas
import datetime

#--------------------------------------------------------------------------#
# 本程序实现使用多个模型进行投票类型的预测，生成未进行颜色映射和进行了颜色影射的图像
# 本程序定义了一系列基于UNet的一系列操作，不是模型本身的定义（nets.unet.py）  
# 使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#--------------------------------------------------------------------------#

class Unet_vote(object):    
    def __init__(self):
        # 更改模型组时以下3个变量可能需要修改
        # 第一组参数，只用前5个模型，权值为5/15,4/15……，结果miou 90.99
        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14"]
        # self.all_models_weights = [0.333,0.267,0.2,0.133,0.067]
        # self.num_models = 5
        
        # 第二组参数，只用前6个模型，权值分别为6/21,5/21……，结果miou 91.00,augmented为91.03※
        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14","final model 02"]
        # self.all_models_weights = [0.2857,0.2381,0.1905,0.1429,0.0952,0.0476]
        # self.num_models = 6
        
        # 第三组参数，只用前7个模型，权值分别为7/28,6/28……，结果miou 91.00，※
        self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14",
                              "final model 02","final model 05"]
        self.all_models_weights = [0.25,0.2143,0.1786,0.1429,0.1071,0.0714,0.0357]
        self.num_models = 7

        # 第四组参数，只用前8个模型，权值分别为8/36,7/36……，结果miou 90.99
        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14",
        #                       "final model 02","final model 05","final model 04"]
        # self.all_models_weights = [0.222,0.194,0.167,0.139,0.111,0.083,0.056,0.028]
        # self.num_models = 8
        
        # 第五组参数，只用前9个模型，权值分别为9/45,8/45……，结果miou 91.03，augmented test为91.05※
        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14",
        #                       "final model 02","final model 05","final model 04","final model 09"]
        # self.all_models_weights = [0.25, 0.2143, 0.1786, 0.1429, 0.1071, 0.0714, 0.0357, 0.0667, 0.0571]
        # self.num_models = 9

        # 第五组参数，只用前9个模型，权值分别为1/6,1/9,1/18(各3项)，结果miou  
        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14",
        #                       "final model 02","final model 05","final model 04","final model 09"]
        # self.all_models_weights = [0.167, 0.167, 0.167, 0.111, 0.111, 0.111, 0.056, 0.056, 0.056]
        # self.num_models = 9

        # 只用前9个模型，权值分别为1/6,1/9,1/18(各3项)，结果miou  
        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14",
        #                       "final model 02","final model 05","final model 04","final model 09","final model 12","final model 13","final model 03"]
        # self.all_models_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # self.num_models = 12

        # 第六组参数，只用前10个模型，权值分别为10/55,9/55……，结果miou 90.99 
        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14",
        #                       "final model 02","final model 05","final model 04","final model 09","final model 12"]
        # self.all_models_weights = [0.182,0.164,0.145,0.127,0.109,0.091,0.073,0.055,0.036,0.018]
        # self.num_models = 10

        # self.all_models_id = ["final model 01","final model 11","final model 10","final model 06","final model 14","final model 02"
        #                       ,"final model 05","final model 04","final model 09"]
        #                     #   ,"final model 12"]
        #                     #   ,"final model 13","final model 03","final model 15","final model 08"
        
        # self.all_models_weights = [9,8,7,6,5,4
        #                             ,3,2,1]
                                    # ,0.0476]
                                #   ,0.0381,0.0286,0.019,0.0095
        # self.all_models_weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,
        #                            0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        # self.all_models_weights = [0.2609,0.2174,0.1739,0.1739,0.0870,0.0870]
        # 第一次在笔记本运行时使用的是这组权重
        # self.all_models_weights = [0.1053,0.1053,0.1053,0.1053,0.0789,0.0789,0.0789,
        #                            0.0789,0.0526,0.0526,0.0526,0.0526,0.0263,0.0263]
        # self.all_models_weights = [0.0714,0.0714,0.0714,0.0714,0.0714,0.0714,0.0714,
        #                            0.0714,0.0714,0.0714,0.0714,0.0714,0.0714,0.0714] 

        self.miou_out_path = "./miou_out/"
        
        self.image_size = [512, 512]
        self.mix_type = 1
        self.num_classes = 9
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
    
    # 将预测的像素类别进行颜色映射
    # original_image是原图像，image是预测后的无颜色图像    
    def nocolor_to_color(self, original_image, image):
        pr = np.array(image) 
        original_image  = cvtColor(original_image)
        orininal_h      = np.array(original_image).shape[0]
        orininal_w      = np.array(original_image).shape[1] 
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(original_image, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(original_image, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        # 为了方便对预测结果进行标记，返回模型的id
        model_id = "voted"
        return image, model_id
    #---------------------------------------------------#
    # 对image多个模型预测（分割），并进行投票获得最终结果
    # 得到颜色映射后的png图像，用于直接查看结果
    #---------------------------------------------------#
    def get_result_colors(self, image):
        original_image = copy.deepcopy(image)
        image_png = self.get_result_nocolor(image)
        # 将无颜色映射的分割结果图像转换成有颜色映射的图像
        image_png, model_id = self.nocolor_to_color(original_image, image_png)
        return image_png, model_id, self.all_models_id, self.all_models_weights

    #---------------------------------------------------#
    # 在miou_out文件夹下找对应模型的预测结果，进行堆叠
    # 使用投票程序获得最终结果，用于进行miou计算，
    # 因为没有进行颜色映射，直接看图片是全黑的
    #---------------------------------------------------#
    def get_result_nocolor(self, image):
        num_models = len(self.all_models_id)
        # 单个模型预测的结果，这里主要是创建该维度的array
        image_voted = np.zeros((self.image_size[0],self.image_size[1]))
        # 多个模型预测结果的堆叠，这里主要是创建该维度的array
        image_multi = np.zeros((num_models,self.image_size[0],self.image_size[1]))
        # 该for循环在miou_out目录下寻找image对应的预测结果（无颜色），并将结果进行堆叠
        for i in range(num_models):
            # 在miou下寻找名称带有self.all_models_id[i]的文件夹
            # 并定位到其子文件夹segmentation-results
            # 读取与image文件名称相同但后缀名为png的文件
            # 只能找到一个，一旦找到，退出下面双重循环
            for root, dirs, files in os.walk(self.miou_out_path):
                for folder in dirs:
                    if self.all_models_id[i] in folder:
                        # 找到包含model_id的文件夹
                        folder_path = os.path.join(root, folder)
                        # 构建segmentation-results文件夹路径
                        segmentation_folder = os.path.join(folder_path, 'segmentation-results')
                        # 检查segmentation-results文件夹是否存在
                        if os.path.exists(segmentation_folder):
                            # print("segmentation_folder存在")
                            # 查找与image文件名称相同但后缀名为png的文件
                            image_base_name, image_ext = os.path.splitext(image.filename)
                            image_base_name = image_base_name.split('/')[3]
                            png_filename = image_base_name + '.png'
                            png_path = os.path.join(segmentation_folder, png_filename)
                            if os.path.exists(png_path):
                                image_png = Image.open(png_path)
                                # print(png_path)
                                break
                break
            image_png = np.array(image_png)
            # 二维单通道数据堆叠成三维
            # print(image_png.shape)
            image_multi[i,:,:] = image_png[:,:]
            # print(image_multi.shape)
        # print("各模型预测结果png堆叠完成")   
         
        # 对单独像素的预测数值进行计数，取出现频次最多的作为作为最后数值
        # 需运行512*512次，耗费时间较长，改用下面单循环的方式
        # for i in range(image_png.shape[0]):
        #     for j in range(image_png.shape[1]):
        #         # 以每个model预测的类别(0-8)作为index，1作为value构建单列的dataframe，列名为w
        #         ke_va = pandas.DataFrame(data=np.ones(self.num_models),index=image_multi[:,i,j], columns=list("w"))
        #         # 调用函数，获得乘以每个模型weight的series(类字典类型)，
        #         # result的index仍然为类别，value为乘以每个模型权值后的值
        #         result = self.weighted_softmax(ke_va["w"], self.all_models_weights)
        #         # 将result从series转化为dataframe，合并key相同的项，value相加
        #         merged_df = result.to_frame().groupby(result.index).sum()
        #         # 取value最高的index，得到的也就是投票后的类别结果
        #         image_voted[i,j] = merged_df['w'].idxmax()
        # image_voted = Image.fromarray(np.uint8(image_voted))
        # # print(image_multi.shape)
        
        # 将image_multi转化为2维，目的是将循环的主操作设在最外层
        # 尽可能进行优化
        im_1d = image_multi.transpose(1,2,0).flatten()
        im_2d = im_1d.reshape(-1,len(self.all_models_weights))
        # print(im_2d.shape)
        # print(image_multi.shape)
        arr = im_2d

        # 创建一个list
        result = []
        element_weights = {}
        # 遍历,
        for i in range(arr.shape[0]):
            element_weights = {}
            for j in range(arr.shape[1]):
                weight = self.all_models_weights[j]
                element = arr[i,j]
                # print(weight)
                # print(element)
                # 如果元素已经在字典中，累加权重
                if element in element_weights:
                    element_weights[element] += weight
                else:
                    # 否则，将元素添加到字典中
                    element_weights[element] = weight
            # print("一行权重操作完成" + str(i))
            max_key = max(element_weights, key=element_weights.get)
            result.append(max_key)
        
        # 对result重整形状，得到投票结果的array形式
        r_size = image_multi.shape[1]
        result = np.array(result).reshape(r_size,r_size) 
        image_voted = Image.fromarray(np.uint8(result))
                
        # print('Total time:' + str(end-start)) 
        # image_voted.save("./img/img01_1loop.png")
        # image_voted.save("./img/img01_2loop.png")        
        return image_voted
    
    # 双重循环投票方式会使用该函数，单循环方式不使用
    def weighted_softmax(self, logits, weights):
        # 计算指数值
        exp_logits = np.exp(logits)
        # print(exp_logits)
        # print(np.sum(exp_logits))
        
        # 加权指数值
        weighted_exp_logits = exp_logits * weights
        # print(weighted_exp_logits)
        
        # 计算加权指数值的和
        sum_weighted_exp = np.sum(weighted_exp_logits)
        # print(sum_weighted_exp)
        
        # 计算带权重的Softmax
        weighted_softmax = weighted_exp_logits / sum_weighted_exp    
    
        return weighted_softmax