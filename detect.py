import argparse
from deep_sort import build_tracker
import torch
from models.selectdevice import*
import torch.backends.cudnn as cudnn
from data.config import VOC_CLASSES
import  torchvision.transforms as T
from deep_sort import build_tracker
from data.config import MULTI_ANCHOR_SIZE
import numpy as np
import cv2
from loss import tools
from video import  *
from PIL import Image,ImageDraw,ImageFont
import random,os
from data.config import DEEPSORT
import shutil

parser = argparse.ArgumentParser(description='YOLONT Detection')
parser.add_argument('-v', '--version', default="YOLONT",
                    help='YOLONT')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC  dataset')
parser.add_argument('-size', '--input_size', default=512, type=int,
                    help='Batch size for training')
parser.add_argument('--trained_model', default='model/YOLONT.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.045, type=float,
                    help='Final confidence threshold')
parser.add_argument("--video",default="",type=str,help="input video")
parser.add_argument("--FILE_Image",default="",type=str,help="input PATH_File")
parser.add_argument("--Image",default="",type=str,help="input Image")
parser.add_argument("--save",default="images/output",type=str,help="save's path")
parser.add_argument("--mask",default=14,type=int,help="mask")
parser.add_argument("--track",default=False,type=bool,help="build track")
parser.add_argument("--delay",default=0,type=int,help="delay time")
args = parser.parse_args()
class tracks():
    def __init__(self):
        pass
    def  get_mask(self,cls):
        a=[args.mask]
        mask=[]
        for i in cls:
            if i in a:
              mask.append(True)
            else:
                mask.append(False)
        return  mask
    def build(self,b=None,s=None,c=None,img=None,w=None,h=None):
        sort=DEEPSORT
        im0=img
        deepsorts = build_tracker(cfg=sort, use_cuda=True)
        boxes,score,cls_inds=b,s,c
        bboxes,confs,ids=[],[],[]
        for j, box in enumerate(boxes):
              if (score[j])> args.visual_threshold:
                        confs.append(score[j])
                        ids.append(cls_inds[j])
                        xywh = int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h)
                        cv2.rectangle(im0,(xywh[0], xywh[1]),(xywh[2],xywh[3]),(255,0,0),3)
                        xy_wh=(xywh[0]+xywh[2])//2,(xywh[1]+xywh[3])//2,xywh[2]-xywh[0],xywh[3]-xywh[1]
                        bboxes.append(xy_wh)
        a,b,c=np.array(bboxes), np.array(confs), np.array(ids)
        mask=self.get_mask(c)
        bbox_xywh = a[mask]
        # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        cls_conf = b[mask]
        # do tracking
        outputs = deepsorts.update(bbox_xywh, cls_conf, im0)
        if len(outputs)!=0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            for inds,i in enumerate(bbox_xyxy):
             cv2.rectangle(im0,(i[0], i[1]),(i[2],i[3]),(255,0,0),3)
             for id in identities:
              pilimg = Image.fromarray(im0)
              draw = ImageDraw.Draw(pilimg) # 图片上打印
              font = ImageFont.truetype("comici.ttf", 27, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
              draw.text((i[0]-50, i[1]-50), "ID:%s"%inds, (0, 255, 60), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
              im0=np.asarray(pilimg)
        cv2.imshow("1",cv2.resize(im0,(512,512)))            
        cv2.waitKey(args.delay)
tracks=tracks()
def inference():
    # get device
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    device=sel()
    # load net
    num_classes = len(VOC_CLASSES)
    input_size = [args.input_size, args.input_size]

    from models.model import YOLONT
    anchor_size = MULTI_ANCHOR_SIZE
    net = YOLONT(device, input_size=input_size, num_classes=len(VOC_CLASSES),trainable=False, anchor_size=anchor_size,train=False)
    
   
    model=torch.load(args.trained_model, map_location=device)
    net.load_state_dict(model)
    net.to(device).eval()
    mean, std = (0.406, 0.456, 0.485), (0.225, 0.224, 0.229)
    def _pre(img=None,input_size=None):
        img_ = cv2.resize(img, (input_size, input_size)).astype(np.float32)
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

    def _net(net, device, input_size, thresh,perdatas=None,out=None,file_image=None,image=None,track=None):
        img_raw, img_, h, w = perdatas
        img_tensor = img_.unsqueeze(0).to(device)
        with torch.no_grad() as grad:
            bboxes, scores, cls_inds = net(img_tensor)
            if args.track:
             track.build(b=bboxes,s=scores,c=cls_inds,img=img_raw,w=w,h=h)
            if not args.track:
                CLASSES = VOC_CLASSES
                class_color = tools.CLASS_COLOR
                for i, box in enumerate(bboxes):
                    cls_indx = cls_inds[i]
                    xmin, ymin, xmax, ymax = box[0]*w,box[1]*h,box[2]*w,box[3]*h
                    if (scores[i])> thresh:
                        cv2.rectangle(img_raw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
                        cv2.rectangle(img_raw, (int(xmin), int(abs(ymin) - 20)), (int(xmax), int(ymin)), class_color[int(cls_indx)],
                                    -1)
                        mess = '%s' % (CLASSES[int(cls_indx)])
                    
                        cv2.putText(img_raw, mess, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow("1",img_raw)
                if out!=None:
                 out.write(img_raw)
                elif file_image!=None:
                 cv2.imwrite(os.path.join(args.save,file_image+'.jpg'),img_raw)
                elif image!=None:
                    print(os.path.join(args.save,image+'.jpg'))
                    cv2.imwrite(os.path.join(args.save,image+'.jpg'),img_raw)
                cv2.waitKey(args.delay)
    def prepare(net=None,device=None):
        if args.video!='':
            print(args.video)
            video=Video(videofile=args.video)
            h,w,c=next(video).shape
            forcc=cv2.VideoWriter_fourcc(*'mpv4')
            out=cv2.VideoWriter(os.path.join(args.save,"out.mp4"),forcc,30,(w,h))
            for image in iter(video):
                _=_pre(img=image,input_size=args.input_size)
                _net(net,device,input_size=args.input_size,thresh=args.visual_threshold,perdatas=_,out=out,track=tracks)
        elif args.FILE_Image!='':
            #r"K:\深度学习数据集\MOT数据集\test(1)\test\MOT20-07\img1"
            file_image=FILE_Image(args.FILE_Image)
            for name,image in iter(file_image):
               _=_pre(img=image,input_size=args.input_size)
               _net(net,device,input_size=args.input_size,thresh=args.visual_threshold,perdatas=_,file_image=name,track=tracks)
        elif args.Image!="":
            image=pictrue_Image(args.Image)
            name=image.path[:-4]
            ids=image.path[:-4].rindex('/')
            name=name[ids+1:]
            _=_pre(img=image.image,input_size=args.input_size)
            _net(net,device=device,input_size=args.input_size,thresh=args.visual_threshold,perdatas=_,image=name)
    prepare(net=net,device=device)

if __name__ == '__main__':
    inference()
