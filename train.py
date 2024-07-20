import os
import datetime
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("start timing: " + str(start))
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda = True
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP，即使是多卡也不需要设置为True。
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     分类的像素类别n+1
    #-----------------------------------------------------#
    num_classes = 9 
    #-----------------------------------------------------#
    #   主干网络选择
    #   vgg or resnet50
    #-----------------------------------------------------#
    backbone    = "vgg"
    # backbone    = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用backbone的预训练权重，在模型构建时加载。
    #                   若设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = False
    #----------------------------------------------------------------
    #------------------------------------------------------------#
    #   一般来讲，随机权值效果较差，一般加载在imagenet上预训练的权值！
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = "./model_data/unet_vgg_voc.pth"
    # model_path  = "./logs/last_epoch_weights.pth"
    # model_path  = "./model_data/unet_resnet_voc.pth"

    #-----------------------------------------------------#
    #   input_shape     输入图片的大小，32的倍数，不建议设置过大
    #-----------------------------------------------------#
    input_shape = [512, 512]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #   加载时会提醒finel部分参数失败，没有关系，因为UNet去掉了原backbone的全连接部分。 
    #   特别注意：总训练轮次是UnFreeze_Epoch, 非冻结轮次是UnFreeze_Epoch - Freeze_Epoch               
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
 
    UnFreeze_Epoch      = 150
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam时建议设置为Init_lr=1e-4
    #                   当使用SGD时建议设置为Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-4                                
    Min_lr              = Init_lr * 0.01                      
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，包括adam和sgd
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   学习率下降方式，'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 2
    
    #------------------------------#
    #   数据集路径
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    # 未增强training data
    # VOCdevkit_path  = 'VOCdevkit_noaugment'
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    #   在矿物图像中各主要类别（闪锌、方铅、黄铜、黄铁等）样本数量基本平衡，
    #   但少数类别（脉石、连生-多类别矿物等）可能样本较少
    #   若为False，就是CrossEntropyLoss的计算
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。需要手动设置，不是可训练的参数
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------------------#
    num_workers     = 2

    #------------------------------------------------------#
    #   设置GPU
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    # print(ngpus_per_node)
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    #----------------------------------------------------#
    #   下载预训练权重
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        # 如果需要加载预训练权重且不是分布式（DDP,LINUX下适用），下载对应的权重（vgg or resnet）
        else:
            download_weights(backbone)
    # train(): 将本层及子层的training设定为True,Unet父类Module的方法
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    # model_path是存放权值文件的路径，在本代码前面部分设置 
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        # 读取预训练的权重，在将要训练的模型（model）中加载
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有加载成功的Key，不影响训练过程
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "logs_" + str(time_str))
        model_dir       = os.path.join(log_dir, 'model')
        model_best_dir  = os.path.join(log_dir, 'model_best')
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    # fp16：半精度浮点数，基于AMP架构，牺牲精度提高显存利用率
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    # 设置model和孩子模型的training参数为True
    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        # 单卡的情况（DP模式），单机双卡且在windows下，也使用这种模式
        else:
            model_train = torch.nn.DataParallel(model, device_ids=[0,1])
            # 确定为单机单卡的使用这条语句
            # model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        # 进行冻结训练
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结backbone部分进行其他部分训练
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()
            
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        # batch_size可能是Freeze_batch_size或者UnFreeze_batch_size，原始设置均为2
        # 设置的越大，对GPU及显存要求就越高
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # 获取训练和验证数据集
        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        # 获取训练和验证集Dataloader
        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)
        
        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        # 写入了一些信息在epoch_miou.txt文件中
        # local_rank可能是与GPU分布式训练相关的参数，应该由系统赋予，通过代码获取  by myself
        # val_lines:val set 列表（val.txt文件内容组成的list）
        # VOCdevkit_path:存放数据集的总目录，默认为项目目录下VOCdevkit
        # log_dir:日志目录，每次训练都会生成，以logs_开头，有时间戳信息
        # eval_flag：评估标志，布尔值，是否在训练中进行评估
        # period：评估标志，代表多少个epoch评估一次
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, train_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        # 具体训练epoch为UnFreeze_Epoch(默认100)-Init_Epoch(默认0)，也就是最大轮次数为UnFreeze_Epoch
        # 若Init_Epoch大于Freeze_Epoch,则不进行冻结训练，直接进行UnFreeze_Epoch-Init_Epoch次非冻结训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   如果有冻结训练（UnFreeze_flag==False, Freeze_Train==True），且当前epoch已大于冻结epoch，
            #   即已进行完冻结训练，将进入（或已经进入了）非冻结训练，则满足条件，执行if后语句
            #   非冻结训练阶段，及所有参数需更新阶段所要进行的设置
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            # 如果分布式执行
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 一次epoch的训练，包括Forward propagation和Backward propagation
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, 
                    focal_loss, cls_weights, num_classes, fp16, scaler, save_period, log_dir, 
                    model_dir, model_best_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
    
    # 显示和记录训练耗时信息
    end = datetime.datetime.now()
    print("end timing: " + str(end))
    print('Total time:' + str(end-start))
    file_name = time.strftime('%Y_%m_%d_%H_%M_%S - training info.txt',time.localtime(time.time()))
    file_name = os.path.join(log_dir, file_name)
    with open (file_name,'w') as f:
        f.write("{:<50}".format("start time:") + str(start) + "\n")
        f.write("{:<50}".format("end time: ") + str(end) + "\n")
        f.write("{:<50}".format("total time: ") + str(end-start) + "\n")
        f.write("{:<50}".format("backbone: ") + str(backbone) + "\n")
        f.write("{:<50}".format("Freeze_Epoch: ") + str(Freeze_Epoch) + "\n")
        f.write("{:<50}".format("UnFreeze_Epoch(total epochs): ") + str(UnFreeze_Epoch) + "\n")
        f.write("{:<50}".format("Freeze_batch_size: ") + str(Freeze_batch_size) + "\n")
        f.write("{:<50}".format("Unfreeze_batch_size: ") + str(Unfreeze_batch_size) + "\n")
        f.write("{:<50}".format("dice_loss: ") + str(dice_loss) + "\n")
        f.write("{:<50}".format("focal_loss: ") + str(focal_loss) + "\n")
        f.write("{:<50}".format("num_workers: ") + str(num_workers) + "\n")
        f.write("{:<50}".format("model_path(loaded weight pth):") + str(model_path) + "\n")
    os.system('shutdown -s -t 60')