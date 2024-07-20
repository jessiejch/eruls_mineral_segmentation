import os
import time

from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 9
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #-------------------------------------------------------------------------#
    count           = False
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["_background_","pyrite","galena","sphalerite","chalcopyrite","bornite","magnetite","pyrrhotite","gangue"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    base_path  = './img/'
    
    # 要参与计算的image文件list，无后缀名，方便后期image和label是名称相同但后缀名不同）
    # image_ids       = open(os.path.join(base_path, "test_txt/test.txt"),'r').read().splitlines() 
    # 测试用，运行./datasets/generate_txt.py生成
    # 如测试集发生变化，重新运行./datasets/generate_txt.py生成txt
    image_ids       = open(os.path.join(base_path, "test_txt/test_augmented.txt"),'r').read().splitlines()

    # label文件存放的目录，如测试集发生变化则需要修改
    gt_dir          = os.path.join(base_path, "512size_test_label_augmented_6x/")
    # gt_dir = "./datasets/test/confusion matrix/label"

    # predicted result dir（预测结果存放路径，miou等计算结果也存在这里）
    model_id        = Unet._defaults.get("model_path").split('-')[1]
    miou_out_path   = "./miou_out/" + model_id[1:] + ' - ' + time.strftime('%Y_%m_%d %H.%M.%S - predict info',time.localtime(time.time()))
    pred_dir        = os.path.join(miou_out_path, 'segmentation-results')
    pred_dir_color  = os.path.join(miou_out_path, 'segmentation-results-color')

    # 获得预测结果
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        
        if not os.path.exists(pred_dir_color):
            os.makedirs(pred_dir_color)
            
        print("Load model.")
        # load model（加载模型）
        unet = Unet()
        print("Load model done.")

        # print("Get predict result.")
        # 读取所有test data中的图像，如测试集发生变化则需要修改
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(base_path, "512size_test_img_augmented_6x/"+image_id+".jpg")
            image       = Image.open(image_path)
            image_bk    = image
            # print("done")
            # 对image进行预测（分割），得到结果png图像，用于进行miou计算，因为没有进行颜色映射，直接看图片是全黑的
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
            
            # 对image进行预测（分割），得到颜色映射后的png图像，用于直接查看结果
            # model_id是返回的预测模型名称，此处暂不使用
            r_image, model_id = unet.detect_image(image_bk, count=count, name_classes=name_classes)
            r_image.save(os.path.join(pred_dir_color, image_id + '_' + model_id + ".png"))
        print("Get predict result done.")
        f_name = os.path.join(miou_out_path,"logs.txt")
        with open (f_name,'w') as f:
            f.write("Model: " + str(unet._defaults['model_path']) + "\n")
            f.write("The total number of predicted image: " + str(len(image_ids)) + "\n")
        
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, arr_IoUs, arr_Recall, arr_class_PA = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        # print(IoUs)
        # print(PA_Recall)
        # print(Precision)
        show_results(miou_out_path, hist, arr_IoUs, arr_Recall, arr_class_PA, name_classes, model_id)