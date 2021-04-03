import torch
gpu_count=torch.cuda.device_count()
def sel():
    device=torch.device('cuda:'+str(0))  if torch.cuda.is_available() else  torch.device('cpu')
    return device

if __name__=='__mian__':
    sel()