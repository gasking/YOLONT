import cv2,os,sys,argparse
import  os.path as osp
class arugement():
  def __init__(self):
    argvs=argparse.ArgumentParser(description="select function")
    argvs.add_argument("--moive",default="",type=str,help="input video")
    argvs.add_argument("--Dir_File",default="",type=str,help="input PATH_File")
    argvs.add_argument("--photo",default="",type=str,help="input Image")
    self.arga=argvs.parse_args()
  def _get(self):
      return self.arga
class Video():
    def __init__(self,videofile):
        cv2.setNumThreads(32)
        self.cap=cv2.VideoCapture(videofile)
        self.toatl=self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.inds=0
    def __next__(self):
     if self.inds<self.toatl:
        t=self.inds
        tf,frame=self.imshow(ind=t)
        self.inds=self.inds+1
        return  frame
     else:
         raise  StopIteration
    def __iter__(self):
        return self
    def imshow(self,ind):
        return self.cap.read()

class  FILE_Image():
    def __init__(self,path):
        self.path=path
        cv2.setNumThreads(32)
        self.file= os.listdir(self.path)
        self.total=len(self.file)
        self.ind=0
    def __iter__(self):
        return self
    
    def __next__(self):
      if self.ind<self.total:
        t=self.ind
        frame=cv2.imread(osp.join(self.path,self.file[t]))
        self.ind=self.ind+1
        name=self.file[t][:-4].replace(self.path,"")

        return self.file[t][:-4], frame
      else:
          raise StopIteration
class pictrue_Image():
    def __init__(self,path):
     self.path=path
     self.image=self.get()
    def get(self):
        return cv2.imread(self.path)
if __name__=="__main__":
    arugements=arugement()._get()
    if arugements.moive!='':
        video=Video(videofile=arugements.moive)
        for image in iter(video):
            cv2.imshow("1",cv2.resize(image,(416,416)))
            cv2.waitKey(1)
    elif arugements.Dir_File!='':
        #r"K:\深度学习数据集\MOT数据集\test(1)\test\MOT20-07\img1"
        Video=FILE_Image(arugements.Dir_File)
        for image in iter(Video):
            cv2.imshow("1",cv2.resize(image,(416,416)))
            cv2.waitKey(1)
    elif arugements.photo!="":
        image=pictrue_Image(arugements.photo)
        cv2.imshow("1",cv2.resize(image,(416,416)))
        cv2.waitKey(1)
