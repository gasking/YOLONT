from data import *
from utils.augmentations import SSDAugmentation
import torch.backends.cudnn as cudnn
import os
import time
import math
import random
from loss import tools
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import argparse
from tqdm import tqdm
from models.selectdevice import  *
from getanchor import get_anchor
from data.config import*
voc=VOC_ROOT
def  pardata():
  train_txt=open(os.path.join(voc,'ImageSets\\Main\\train.txt'),'w+')
  test_txt=open(os.path.join(voc,'ImageSets\\Main\\test.txt'),'w+')
  file=os.listdir(os.path.join(voc,'Annotations'))
  random.shuffle(file)
  train_file=file[:int(len(file)*0.7)]
  test_file=file[int(len(file)*0.7)+1:]
  for ind,files in enumerate(train_file):
        filename=os.path.join(voc,'Annotations',files)
        if ind==len(train_file)-1:
            train_txt.write(filename[:-4].replace(os.path.join(voc,'Annotations')+'\\',''))
        else:
            train_txt.write(filename[:-4].replace(os.path.join(voc,'Annotations')+'\\','')+'\n')
  for ind,files in enumerate(test_file):
        filename=os.path.join(voc,'Annotations',files)
        if ind==len(test_file)-1:
            test_txt.write(filename[:-4].replace(os.path.join(voc,'Annotations')+'\\',''))
        else:
            test_txt.write(filename[:-4].replace(os.path.join(voc,'Annotations')+'\\','')+'\n')
  train_txt.close()
  test_txt.close()
pardata()
cfg = voc_ab
def parse_args():
    parser = argparse.ArgumentParser(description='YOLONT Detection')
    parser.add_argument('-v', '--version', default='YOLONT',
                        help='YOLONT')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC or COCO dataset')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--dataset_root', default=VOC_ROOT, 
                        help='Location of VOC root directory')
    parser.add_argument('--num_classes', default=cfg['num_classes'], type=int,
                        help='The number of dataset classes')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=True,
                        help='use multi-scale trick')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--save_folder', default='pth/', type=str,
                        help='Gamma update for SGD')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--resume', type=str, default=None,
                        help='fine tune the model trained on MSCOCO.')


    return parser.parse_args()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    hr = False
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True

    cfg = voc_ab

    device = sel()



    # use multi-scale trick
    if args.multi_scale:
        print('use multi-scale trick.')
        input_size = [640, 640]
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(input_size, mean=(0.406, 0.456, 0.485),
                                                                                 std=(0.225, 0.224, 0.229)))

    else:
        input_size = cfg['min_dim']
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], mean=(0.406, 0.456, 0.485),
                                                         std=(0.225, 0.224, 0.229)))

    # build model
    if args.version == 'YOLONT':
        from models.model import YOLONT
        #自定义数据锚点
        anchor_size=get_anchor()
        #voc数据集锚点
        #anchor_size = MULTI_ANCHOR_SIZE
        yolo_net = YOLONT(device, input_size=input_size, num_classes=args.num_classes, trainable=True,
                          anchor_size=anchor_size, hr=hr)
    else:
        print('Unknown version !!!')
        exit()
    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log_path = os.path.join('log\\voc\\', args.version, "log")
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    model = yolo_net
    model.to(device)

    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    epoch_size = len(dataset) // args.batch_size
    max_epoch = cfg['max_epoch']

    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    t0 = time.time()

    # start training
    for epoch in range(max_epoch):

        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5 * (base_lr - 0.00001) * (
                        1 + math.cos(math.pi * (epoch - 20) * 1. / (max_epoch - 20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)

        # use step lr
        else:
            if epoch in cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(data_loader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)

            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                input_size = [size, size]
                model.set_grid(input_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=False)

            # make train label
            targets = [label.tolist() for label in targets]
            targets = tools.multi_gt_creator(input_size, yolo_net.stride, targets, anchor_size=anchor_size)
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)
            # backprop and update
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)

                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch + 1, max_epoch, iter_i, epoch_size, tmp_lr,
                         conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), input_size[0],
                         t1 - t0),
                      flush=True)
                t0 = time.time()

                # change input dim
                # But this operation will report bugs when we use more workers in data loader, so I have to use 0 workers.
                # I don't know how to make it suit more workers, and I'm trying to solve this question.
                # data_loader.dataset.reset_transform(SSDAugmentation(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save,
                                                        args.version + '_' + repr(epoch + 1) + '.pth')
                       )


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
