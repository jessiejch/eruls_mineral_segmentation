## Ensemble Res-UNet Learning System(ERULS) for semantic segmentation of optical microscopy image 
---

### 训练步骤
#### 训练数据集
1、使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。   
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、注意修改train.py的num_classes为分类个数+1。    
6、运行train.py即可开始训练。  
7、datasets下目录需自行建立。可使用自己的数据集，正确设置格式即可。

### 预测步骤
#### 一、使用预训练权值
  如果想要利用预训练的模型进行预测，先下载权值，再放入model_data，在train.py中正确设置(具体参看代码内注释)，运行即可。  

#### 二、使用自己训练的权重模型
1. 按照训练步骤训练。    
2. 在unet.py文件里面，在如下部分修改model_path、backbone和num_classes使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    

### 评估步骤（获得各项评估指标数值）
1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别名称。  
3、运行get_miou.py可使用单个模型对测试数据集进行预测，获得各项数据指标。  
4、运行get_miou_vote.py可使用ERULS对测试数据集进行预测，获得各项数据指标。集成的模型和权值需要事先在unet_vote.py中设置好。

## Reference
https://github.com/bubbliiiing/unet-pytorch  
https://github.com/ggyyzm/pytorch_segmentation
