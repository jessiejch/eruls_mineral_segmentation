import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            # 加入smooth train loss 和smooth val loss
            # plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            # if self.val_loss_flag:
                # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.yticks(np.arange(0,1,0.1))
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    # 在train.py中创建了实例，传参进行初始化
    def __init__(self, net, input_shape, num_classes, image_ids, train_image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.train_image_ids    = [train_image_id.split()[0] for train_image_id in train_image_ids]
        self.mious      = [0]
        self.pa_racall  = [0]
        self.precision  = [0]
        self.test_mious = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        # self.period来源于在train.py中初始化EvalCallback类实例时设置的参数（最后一个）
        # 等于在train.py中设置的参数eval_period（原始设置为5，可自行修改）
        # self.image_ids是验证集文件名列表，self.train_image_ids是训练集文件名列表
        # 此处计算train + val set的miou
        # train_val_ids = self.image_ids + self.train_image_ids
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")

            # 逐一读取图像
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                # 整合成完整路径
                image_path  = os.path.join(self.dataset_path, "JPEGImages/"+image_id+".jpg")
                # 打开image图像文件
                image       = Image.open(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate  miou.")
            _, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100
            temp_recall = np.nanmean(PA_Recall) * 100
            temp_precision = np.nanmean(Precision) * 100

            self.mious.append(temp_miou)
            self.pa_racall.append(temp_recall)
            self.precision.append(temp_precision)

            self.epoches.append(epoch)
            
            # 将compute_mIoU计算的结果写入txt文件
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")

            with open(os.path.join(self.log_dir, "epoch_recall.txt"), 'a') as f:
                f.write(str(temp_recall))
                f.write("\n")

            with open(os.path.join(self.log_dir, "epoch_mpa.txt"), 'a') as f:
                f.write(str(temp_precision))
                f.write("\n")
            
            # 绘制miou,recall,precision图 
            plt.figure()
            plt.plot(self.epoches, self.mious, 'r', linewidth = 2, label='train&val set miou')
            plt.plot(self.epoches, self.pa_racall, 'b', linewidth = 2, label='val set recall')
            plt.plot(self.epoches, self.precision, 'g', linewidth = 2, label='val set mpa')
            plt.grid(True)
            plt.yticks(np.arange(0,110,10))
            plt.xlabel('Epoch')
            plt.ylabel('Val Metrics')
            plt.title('Metrics Curve')
            plt.legend(loc="lower right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou_recall_mpa.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            # 递归删除某文件夹下的所有文件和文件夹
            shutil.rmtree(self.miou_out_path)
