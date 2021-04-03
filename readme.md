# 需要下载YOLONT权重文件放在model文件夹下                                   链接: https://pan.baidu.com/s/1vTLu54RWcs4ZrH-Hy-cLxA 提取码: xxbx
# 需下载ckpt.t7目标追踪权重 放在deep_sort\deep\checkpoint 文件夹下          链接: https://pan.baidu.com/s/173pB2t1U0flgn2d1JaT_zA 提取码: ermm

# 1. 推理文件 直接运行.detectpy
   # $python detect.py --Image  file.jpg  # image 
   #    --video  file.mp4  # video
   #    --FILE_Image ./dir  # directory
      
# To run inference on examples in the `images/output` folder:
# bash
# $ python detect.py --FILE_Image ./inference/images/ --trained_model model/YOLONT.pth --visual_threshold 0.02
# ![Image](http://github.com/gasking/YOLONT/raw/master/images/output/bus.jpg)
# ![Image](http://github.com/gasking/YOLONT/raw/master/images/output/kite.jpg)
# 2. 训练需修改config里面的配置
# VOC_ROOT=r'E:\YOLONT\VOCYOLO'
# voc_ab = {
#    'num_classes': 20,
#    'lr_epoch': (300, 650), # (60, 90, 160),
#    'max_epoch': 800,
#    'min_dim': [416, 416],
#    'name': 'VOC'
# } 
# 修改VOC_ROOT 为你的数据的路径 为Annotations、JPEGImages、ImageSets的上层路径，如下所示 VOC2007
# example：
#  VOC2007:
#     Annotations
#       *.xml 存放你的xml文件
#     JPEGImages
#       *.jpg  *.png  *.bmp  存放你的图片文件
#    ImageSets
#       Main
#        train.txt
#        test.txt
#    train和test里面都是存放文件的命名 
# 修改你要训练的类别个数 不包括背景
#  num_classes : X 训练类别数
#  lr_epoch: (300, 650) 学习率迭代区间
#  max_epoch: 800       最大的epoch 
# config文件中的VOC_CLASS 是以元组的形式存在
#  VOC_CLASSES = (  # always index 0
#    'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#    'cow', 'diningtable', 'dog', 'horse',
#    'motorbike', 'person', 'pottedplant',
#    'sheep', 'sofa', 'train', 'tvmonitor') 
# 运行train_voc.py

# track.py是YOLONT+deepsort的目标检测+目标追踪 demo
#   增加注意力机制
#   ![Image](http://github.com/gasking/YOLONT/raw/master/images/output/track.png)
# 精度
 # 分辨率     MAP(ap50)   flops/浮点运算次数 params/参数个数 model_size
 # 416X416     52.5%    1493393343            3571143b      13.7M
 # 512X512     60.05%   2262181632            3571143b      13.7M
 # 618X618     58.9%    3190029567            3571143b      13.7M
 

