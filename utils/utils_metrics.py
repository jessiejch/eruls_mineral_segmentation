import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 功能：产生 n*n confusion matrix
# a:label(ground-truth,1 dimension)， b:predicted result（1 dimension）, n:number of class
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a形状(H×W,)；b形状(H×W,)
    #   k为掩膜，去除了255这些点（即label中的白色轮廓），a>=0是为了防止bincount()函数出错
    #   mask得到的是一个n*n的全为True的数组
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，斜对角线上的为分类正确的像素点
    #   bincount用于统计数组内每个非负整数的个数
    #   本句为计算confusion matrix的核心代码
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

# 对于二维array，hist.sum(0)表示每列(每行相同位置)数值相加，得到一维array，
# hist.sum(1)表示每行(每列相同位置)数值相加，得到一维array
# np.maximum(array,1)表示array逐位和1进行比较,选择较大值.因为此处array每个数据都是大于1的，所以最终还是array本身
# 以下三个计算metrics的函数，返回的是每一个类别的值，是一个array

# 返回每个类别的Iou
def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

# hist.sum(1) hist每一行累加，得到一维array，返回每个类别的recall
def per_class_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

# hist.sum(0) hist每一列累加，得到一维array
# 表示每个类别的PA，confusion matrix中每个对角线的元素（预测正确），除以当前列的所有元素之和（预测为该类）
# 所有类别的TP/(TP+FP)，之后求均值就得到了mPA
def per_class_PixelAccuracy(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

# 返回一个数值，表示通用指标中的Accuracy，也表示语义分割指标中的PA，
# 表示confusion matrix中正确分类的像素（对角线）与所有像素（矩阵所有数值之和）的比值
def Pixel_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def Frequency_Weighted_Intersection_over_Union(hist):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(hist, axis=1) / np.sum(hist)
        iu = np.diag(hist) / (
                np.sum(hist, axis=1) + np.sum(hist, axis=0) -
                np.diag(hist))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

# png_name_list是所有不带后缀的图像名称？
# gt_dir存放ground-truth的目录
# pred_dir存放预测结果的目录
# num_classes类别数量

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    # png_name_list是所有图像名称列表（无后缀名）
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    # print(gt_dir)
    # print(pred_dir)

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果（模型预测结果），转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签（ground-truth），转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对一张图片计算num_classes×num_classes的hist矩阵，并累加
        #   label为ground-truth，pred为预测值
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        # if内print未在结果中显示？
        if name_classes is not None and ind > 0 and ind % 1000 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, len(gt_imgs),100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PixelAccuracy(hist)),
                    100 * Pixel_Accuracy(hist))
            )
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU、recall和PA值
    #------------------------------------------------#
    arr_IoUs        = per_class_iu(hist)
    arr_Recall      = per_class_Recall(hist)
    arr_class_PA    = per_class_PixelAccuracy(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    # 如果name_classes是None（默认值），条件不满足，不执行for循环
    # 如果name_classes是具体的类别名称，条件满足，执行for循环
    if name_classes is not None:
        for ind_class in range(num_classes):
            # print内容未在结果中显示
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(arr_IoUs[ind_class] * 100, 2)) \
                + '; Recall -' + str(round(arr_Recall[ind_class] * 100, 2))+ '; Pixel Accuracy-' + str(round(arr_class_PA[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(arr_IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(arr_class_PA) * 100, 2)) + '; mRecall: ' + str(round(np.nanmean(arr_Recall) * 100, 2)))  
    # 可能是该行报错导致5个epoch后就停止，因为版本更迭np.int过期了，
    # 将np.int改为np.int32（报错提示也可改为np.int64）   
    return np.array(hist, np.int32), arr_IoUs, arr_Recall, arr_class_PA

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    # fig     = plt.gcf()
    fig = plt.figure(figsize=(8.0, 5.0)) 
    axes    = plt.gca()
    # vgg模型预测结果设置color为brown
    # resnet50模型预测结果设置color为darkblue
    plt.barh(range(len(values)), values, color='brown')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}%".format(val*100)
        t = plt.text(val, i, str_val, color='darkblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

# 保留背景项统计
def show_results(miou_out_path, hist, arr_IoUs, arr_Recall, arr_class_PA, name_classes, model_id, tick_font_size = 12):
    draw_plot_func(arr_IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(arr_IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))
    
    draw_plot_func(arr_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(arr_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(arr_class_PA, name_classes, "mPA = {0:.2f}%".format(np.nanmean(arr_class_PA)*100), "mPA", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    

    with open(os.path.join(miou_out_path, "metrics.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        # 若计算时去除背景，可用下段注释掉的代码
        # moiu_list = []
        # recall_list = []
        # precision_list = []
        # # 改变各list的数据顺序，与结果表格（按训练集中标注的类别数量升序排列）中的一致
        # sort_key = [0,5,7,6,3,2,8,1,4]

        # # 在每个列表中加入排序key
        # for i in range(9):
        #     moiu_list.append((arr_IoUs[i],sort_key[i]))
        #     recall_list.append((arr_Recall[i],sort_key[i]))
        #     precision_list.append((arr_class_PA[i],sort_key[i])) 

        # # 按照key对每个列表进行排序
        # moiu_list.sort(key = second_item)
        # recall_list.sort(key = second_item)
        # precision_list.sort(key = second_item)

        # # 删除列表中的key
        # for i in range(9):
        #     moiu_list[i] = (moiu_list[i][0])
        #     recall_list[i] = (recall_list[i][0])
        #     precision_list[i] = (precision_list[i][0])
       
        # moiu_list = [ round(i*100,2) for i in moiu_list]
        # recall_list = [ round(i*100,2) for i in recall_list]
        # precision_list = [ round(i*100,2) for i in precision_list]
        # moiu_list[0] = 1
        # recall_list[0] = 2
        # precision_list[0] = 3
        f1_score = np.round(2*np.nanmean(arr_Recall)*100*np.nanmean(arr_class_PA)*100/(np.nanmean(arr_Recall)*100+np.nanmean(arr_class_PA)*100),2)

        writer.writerow(name_classes)
        writer.writerow(arr_IoUs)
        writer.writerow(arr_Recall)
        writer.writerow(arr_class_PA)

        writer.writerow(["2:IoUs, 3:Recall, 4:class_PA"])
        writer.writerow(["average","model_id", "IoUs","Recall","class_PA"])
        writer.writerow([0,model_id,np.mean(arr_IoUs),np.mean(arr_Recall),np.mean(arr_class_PA)])
        writer.writerow(["F1_score:",f1_score])
        writer.writerow(["Frequency_Weighted_Intersection_over_Union:",Frequency_Weighted_Intersection_over_Union(hist)])
        

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
    # FWIOU,频权交并比的计算，后期考虑加入到图像结果中
    print(Frequency_Weighted_Intersection_over_Union(hist))
    show_confusion_matrix(hist)

#去掉背景项统计 
# def show_results(miou_out_path, hist, arr_IoUs, arr_Recall, arr_class_PA, name_classes, model_id, tick_font_size = 12):
#     draw_plot_func(arr_IoUs[1:], name_classes[1:], "mIoU = {0:.2f}%".format(np.nanmean(arr_IoUs[1:])*100), "Intersection over Union", \
#         os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = False)
#     print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))
    
#     draw_plot_func(arr_Recall[1:], name_classes[1:], "mRecall = {0:.2f}%".format(np.nanmean(arr_Recall[1:])*100), "Recall", \
#         os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
#     print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

#     draw_plot_func(arr_class_PA[1:], name_classes[1:], "mPA = {0:.2f}%".format(np.nanmean(arr_class_PA[1:])*100), "mPA", \
#         os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
#     print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    

#     with open(os.path.join(miou_out_path, "metrics.csv"), 'w', newline='') as f:
#         writer          = csv.writer(f)
#         # 去掉background类别统计
#         moiu_list = arr_IoUs[1:]
#         recall_list = arr_Recall[1:]
#         precision_list = arr_class_PA[1:]
               
#         moiu_list = [ round(i*100,2) for i in moiu_list]
#         recall_list = [ round(i*100,2) for i in recall_list]
#         precision_list = [ round(i*100,2) for i in precision_list]

#         f1_score = np.round(2*np.nanmean(recall_list)*100*np.nanmean(precision_list)/(np.nanmean(recall_list)*100+np.nanmean(precision_list)*100),2)

#         writer.writerow(name_classes[1:])
#         writer.writerow(moiu_list)
#         writer.writerow(recall_list)
#         writer.writerow(precision_list)

#         writer.writerow(["2:IoUs, 3:Recall, 4:class_PA"])
#         writer.writerow(["average","model_id", "IoUs","Recall","class_PA","F1_score:"])
#         writer.writerow([0,model_id,np.mean(moiu_list),np.mean(recall_list),np.mean(precision_list),f1_score])
#         writer.writerow(["Frequency_Weighted_Intersection_over_Union:",Frequency_Weighted_Intersection_over_Union(hist)])        

#     with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
#         writer          = csv.writer(f)
#         writer_list     = []
#         writer_list.append([' '] + [str(c) for c in name_classes])
#         for i in range(len(hist)):
#             writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
#         writer.writerows(writer_list)
#     print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
#     # FWIOU,频权交并比的计算，后期考虑加入到图像结果中
#     print(Frequency_Weighted_Intersection_over_Union(hist))
#     show_confusion_matrix(hist)

def second_item(item):
        return item[1]

def show_confusion_matrix(hist): 
    classes = ["_background_","pyrite","galena","sphalerite","chalcopyrite","bornite",
                   "magnetite","pyrrhotite""gangue"]
    confusion_matrix = np.array(hist, dtype=np.int) #输入混淆矩阵
    # 计算得到的比例，小数格式，如0.9787，含义是召回率
    proportion=[]
    for i in confusion_matrix:
        for j in i:
            temp=j/(np.sum(i))
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)
    # 处理后的百分比形式，后期可显示在confusion matrix中
    pshow=[]
    for i in proportion:
        pt="%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion=np.array(proportion).reshape(9,9)  # reshape(列的长度，行的长度)
    # pshow=np.array(pshow).reshape(11,11)
    # print(pshow)
    config = {
        "font.family":'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)

    # 显示混淆矩阵，有保存的表格，图像可以在论文中使用
    # plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵 
    #             # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    #             # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes,fontsize=9, rotation=50)
    # plt.yticks(tick_marks, classes,fontsize=9)
    
    # thresh = confusion_matrix.max() / 2.
    #iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    # iters = np.reshape([[[i,j] for j in range(9)] for i in range(9)],(confusion_matrix.size,2))
    # for i, j in iters:
    #     if(i==j):
    #         plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9,color='white',weight=5)  # 显示对应的数字
    #         # plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=6,color='white')
    #     else:
    #         plt.text(j, i-0.12, format(confusion_matrix[i, j]),va='center',ha='center',fontsize=9)   #显示对应的数字
    #         # plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=6)
    
    # plt.ylabel('True label',fontsize=12)
    # plt.xlabel('Predict label',fontsize=12)
    # plt.tight_layout()
    # plt.show() 