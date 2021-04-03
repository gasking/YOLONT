# config.py

IGNORE_THRESH = 0.5
#这个就是数据集的路径
VOC_ROOT=r'F:\PP_AI\Mydemo\Pro\quexian\PP_Road'
voc_ab = {
    'num_classes': 1,
    'lr_epoch': (300, 650), # (60, 90, 160),
    'max_epoch': 800,
    'min_dim': [416, 416],
    'name': 'VOC'
}

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


#注意用元组的形式储存
#VOC_CLASSES =('d00','d10','d20','d40')


# multi level anchor box config for VOC and COCO
# yolo_v3

[ [400.18, 113.37], 
  [175.45, 189.76],
  [392.42, 50.22],
  [372.06, 362.47], 
  [187.24, 32.17], 
  [104.91, 401.97], 
  [151.73, 82.56], 
  [48.18, 403.06], 
  [51.12, 169.84]]
MULTI_ANCHOR_SIZE = [[30.65, 39.12],   [50.3, 102.62],   [94.98, 64.55],
                     [93.5, 177.51],   [165.25, 113.85], [161.83, 240.95],
                     [304.64, 150.34], [251.28, 306.53], [369.38, 261.55]]
#
# MULTI_ANCHOR_SIZE_COCO = [[11.89, 14.24],   [30.14, 35.62],   [45.99, 87.04],
#                           [92.23, 44.43],   [130.78, 99.73],  [78.99, 170.81],
#                           [290.39, 123.89], [165.27, 233.33], [332.57, 279.8]]
#
# # tiny yolo_v3
# TINY_MULTI_ANCHOR_SIZE = [[33.94, 46.2],    [83.07, 94.45],   [107.95, 191.32],
#                           [251.06, 127.57], [191.95, 262.17], [339.96, 268.11]]
#
# TINY_MULTI_ANCHOR_SIZE_COCO = [[14.28, 17.23], [44.4, 46.83],   [68.93, 117.79],
#                                [178.26, 78.7], [144.84, 207.3], [316.17, 245.71]]

DEEPSORT={
  'REID_CKPT': "./deep_sort/deep/checkpoint/ckpt.t7",
  'MAX_DIST': 0.2,
  'MIN_CONFIDENCE': 0.001,
  'NMS_MAX_OVERLAP': 0.3,
  'MAX_IOU_DISTANCE': 0.3,
  'MAX_AGE': 70,
  'N_INIT': 3,
  'NN_BUDGET': 100
  }