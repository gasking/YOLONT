import argparse
import torch
import torch.backends.cudnn as cudnn
from data.config import VOC_CLASSES
import  torchvision.transforms as T
from deep_sort import build_tracker
from data.config import MULTI_ANCHOR_SIZE
import numpy as np
import cv2
from  models.selectdevice import  *
from loss import tools
from PIL import Image,ImageDraw,ImageFont
import cv2 as cv
from deep_sort import build_tracker
import random,os
from video import *
parser = argparse.ArgumentParser(description='YOLONT Detection')
parser.add_argument('-v', '--version', default='YOLONT',
                    help='YOLONT')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC  dataset')
parser.add_argument('-size', '--input_size', default=512, type=int,
                    help='Batch size for training')
parser.add_argument('--trained_model', default='model/YOLONT.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.02, type=float,
                    help='Final confidence threshold')

parser.add_argument('--mask', default=14,type=int
                    ,help='a=mask')

args = parser.parse_args()
def pull_item(img_path=None,stream=None):
    if  img_path==None:
      img =stream
    elif  isinstance(img_path,str):
        img=cv2.imread(img_path)
    else:
        print("\033[30;35m 退出............ \033[0m")
    mean, std = (0.406, 0.456, 0.485), (0.225, 0.224, 0.229)
    img_ = cv2.resize(img, (args.input_size, args.input_size)).astype(np.float32)
    img_ /= 255.
    img_ -= mean
    img_ /= std
    img_ = np.transpose(img_, (0, 1, 2))
    t = T.Compose(
        [
            T.ToTensor()
        ]
    )
    img_tensor = t(img_)
    height, width, channels = img.shape
    return img, img_tensor, height, width
def _pp(path=None,stream=None):   
    device=sel()
    # load net
    num_classes = len(VOC_CLASSES)
    input_size = [args.input_size, args.input_size]
    from models.model import YOLONT
    anchor_size = MULTI_ANCHOR_SIZE
    net = YOLONT(device, input_size=input_size, num_classes=20,trainable=False, anchor_size=anchor_size,train=False)
    model=torch.load(args.trained_model, map_location=device)
    net.load_state_dict(model)
    net.to(device).eval()
    # evaluation
    def _p(net, device, input_size, thresh,path=None,stream=None):
        img_raw, img_, h, w = pull_item(path,stream)
        img_tensor = img_.unsqueeze(0).to(device)
        with torch.no_grad() as grad:
            bboxes, scores, cls_inds = net(img_tensor)
            return bboxes,scores,cls_inds,h,w
    return _p(net, device, input_size,thresh=args.visual_threshold,path=path,stream=stream)
def  get_mask(b):
    a=[args.mask]
    mask=[]
    for i in b:
        if i in a:
           mask.append(True)
        else:
            mask.append(False)
    return  mask
def detect(cfg=None):
    thresh=0.01
    deepsort = build_tracker(cfg, use_cuda=True)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(20)]
    # Run inference
    CLASSES = VOC_CLASSES
    
    path=r'K:\深度学习数据集\MOT数据集\test(1)\test\MOT20-07\img1'
    for i in os.listdir(path=path):
        file=os.path.join(path,i)
        im0=cv2.imread(file)
        cv.imshow("1",cv2.resize(im0,(416,416)))
        boxes,score,cls_inds,h,w=_pp(path=file)
        bboxes,confs,ids=[],[],[]
        for j, box in enumerate(boxes):
              if (score[j])> thresh:
                        confs.append(score[j])
                        ids.append(cls_inds[j])
                      
                        xywh = int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h)
                        label = '%s %.2f' % (CLASSES[int(cls_inds[j])], score[j])
                        
                        xy_wh=(xywh[0]+xywh[2])//2,(xywh[1]+xywh[3])//2,xywh[2]-xywh[0],xywh[3]-xywh[1]
                        bboxes.append(xy_wh)
        a,b,c=np.array(bboxes), np.array(confs), np.array(ids)
        mask=get_mask(c)
        bbox_xywh = a[mask]
        # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        cls_conf = b[mask]
        # do tracking
        outputs = deepsort.update(bbox_xywh, cls_conf, im0)
       
        if len(outputs)!=0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            for inds,i in enumerate(bbox_xyxy):
             cv2.rectangle(im0,(i[0], i[1]),(i[2],i[3]),(255,0,0),3)
             for id in identities:
              #im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
              pilimg = Image.fromarray(im0)
              draw = ImageDraw.Draw(pilimg) # 图片上打印
              font = ImageFont.truetype("comici.ttf", 27, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
              draw.text((i[0]-50, i[1]-50), "ID:%s"%inds, (0, 255, 60), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
              im0=np.asarray(pilimg)
             cv2.imshow("1",cv2.resize(im0,(512,512)))            
             cv2.waitKey(1)  
def  main():
    from data.config import DEEPSORT
    deepsort=DEEPSORT
    detect(cfg=deepsort)
main()


  