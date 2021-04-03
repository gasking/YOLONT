import numpy as np
import cv2
import math
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
global data

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def  lengh(x1,y1,x2,y2):
    return  math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
class  xybox(object):
    #参见汽车尺度 約2.6左右
    def __init__(self):
        self.size=2.5
        self.data=dict()
        self.ids=dict()
        self.threthsod=2#在这里我们初始化为0
        self.init=0
        self.speed=0
        self.che=dict()

        #print("CCCCCCCCCCCC")
        #print("数据类型：",type(self.data))
    def draw_boxes(self,img, bbox, identities=None, offset=(0,0),times=None):
     self.init=self.init+1
     for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]

        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        print("id: ",id)
        print("所有键: ",self.data.keys())

        if id in self.data.keys():
            #print("AAAAAAAAAAAAAAAAAAAAAAAA")
            #lens=((self.data[id][2]-self.data[id][1])/self.size)*lengh((self.data[id][0]+self.data[id][2])/2,(self.data[id][1]+self.data[id][3])/2,(x1+x2)/2,(y1+y2)/2)
            lens =(lengh(
                (self.data[id][0] + self.data[id][2]) / 2, (self.data[id][1] + self.data[id][3]) / 2, (x1 + x2) / 2,
                (y1 + y2) / 2))/(times)*3.6/self.size

            self.data[id]=x1, y1, x2, y2

            if lens==0:
                self.ids[id]=0
            else:
               #if id not in self.ids.keys():
                self.ids[id]=lens
            print("距离:",lens)
        #     w,h=cv2.phaseCorrelate(img[data[id]],img[x1:x2,y1:y2])
        #     #ax1,ay1,ax2,ay2=x1-data[id][0],y1-data[id][1],x2-data[id][2],y2-data[id][3],
        #     print("水平=%d 垂直=%d"%(w,h))
        else:
          print(".............................")
          self.data[id]=x1,y1,x2,y2
        #
        # cv2.imshow("test",img[y1:y2,x1:x2])
        # cv2.waitKey(0)
        # print(data)
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        if id in self.ids.keys():
         self.speed='{}{:g}'.format("", round(self.ids[id],2))
         #self.che[self.ids[id]]=self.speed
        else:
            continue
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        # if self.init==self.threthsod:
        #     self.init=0
        #     sum1=sum(self.che)/self.threthsod
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)  # 这个是我们的框
        #
        #     cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        #     cv2.putText(img, label + '      ' +str(sum1)+ 'KM/H', (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,
        #                 2, [0, 0, 0], 2)
        # #else:
        #    # cv2.putText(img, label + '      ' + self.speed + 'KM/H', (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,
        #          #       2, [0, 0, 0], 2)
        # else:
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)#这个是我们的框

        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        #cv2.putText(img,"id:"+label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,
                    # 2, [0, 0, 0], 2)
        cv2.putText(img,label+'      '+ self.speed+'KM/H',(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,0], 2)
     return img, self.speed



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
