import numpy as np
from data import *
import torch.nn as nn
import torch.nn.functional as F
import math
CLASS_COLOR = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(len(VOC_CLASSES))]
# We use ignore thresh to decide which anchor box can be kept.
ignore_thresh = IGNORE_THRESH
from torch.autograd import Variable
from torch import nn
import torch
from torch.nn import functional as F
class BCELoss(nn.Module):
    def __init__(self,  weight=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, mask):
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = -pos_id * (targets * torch.log(inputs + 1e-14) + (1 - targets) * torch.log(1.0 - inputs + 1e-14))
        neg_loss = -neg_id * torch.log(1.0 - inputs + 1e-14)
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


class MSELoss(nn.Module):
    def __init__(self,  weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, mask):
        # We ignore those whose tarhets == -1.0.
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss



class focal_loss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=20, alpha=None, gamma=2, size_average=True):
        super(focal_loss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha =alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)

        ids = targets.view(-1, 1)
        ids=ids.to(dtype=torch.uint8)
        print("ids:",ids.dtype)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class BCE_focal_loss(nn.Module):
    def __init__(self,  weight=None, gamma=0.3, reduction='mean'):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        pos_id = (targets==1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * (1.0-inputs)**self.gamma * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * (inputs)**self.gamma * torch.log(1.0 - inputs + 1e-14)

        if self.reduction == 'mean':
            return torch.mean(torch.sum(pos_loss+neg_loss, 1))
        else:
            return pos_loss+neg_loss


def generate_anchor(input_size, stride, anchor_scale, anchor_aspect):
    """
        The function is used to design anchor boxes by ourselves as long as you provide the scale and aspect of anchor boxes.
        Input:
            input_size : list -> the image resolution used in training stage and testing stage.
            stride : int -> the downSample of the CNN, such as 32, 64 and so on.
            anchor_scale : list -> it contains the area ratio of anchor boxes. For example, anchor_scale = [0.1, 0.5]
            anchor_aspect : list -> it contains the aspect ratios of anchor boxes for various anchor area.
                            For example, anchor_aspect = [[1.0, 2.0], [3.0, 1/3]]. And len(anchor_aspect) must
                            be equal to len(anchor_scale).
        Output:
            total_anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    """
    assert len(anchor_scale) == len(anchor_aspect)
    h, w = input_size
    hs, ws = h // stride, w // stride
    S_fmap = hs * ws
    total_anchor_size = []
    for ab_scale, aspect_ratio in zip(anchor_scale, anchor_aspect):
        for a in aspect_ratio:
            S_ab = S_fmap * ab_scale
            ab_w = np.floor(np.sqrt(S_ab))
            ab_h =ab_w * a
            total_anchor_size.append([ab_w, ab_h])
    return total_anchor_size

"""
使用ciou替换iou
"""
def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = np.minimum(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = np.maximum(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = np.maximum(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = np.minimum(bboxes1[:, :2],bboxes2[:, :2])

    inter = np.clip((inter_max_xy - inter_min_xy), a_min=0,a_max=np.nan)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer =np.clip((out_max_xy - out_min_xy), a_min=0,a_max=np.nan)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
   # with torch.no_grad():
    arctan =np.arctan(w2 / h2) - np.arctan(w1 / h1)

    v = (4 / (math.pi ** 2)) * np.power((np.arctan(w2 / h2) - np.arctan(w1 / h1)), 2)
    S = 1 - iou
    alpha = v / (S + v)
    w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = np.clip(cious,a_min=-1.0,a_max = 1.0)
    if exchange:
        cious = cious.T
    return cious
def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [xmin, ymin, xmax, ymax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


def multi_gt_creator(input_size, strides, label_lists=[], anchor_size=None):
    """creator multi scales gt"""
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h, w = input_size
    num_scale = len(strides)
    gt_tensor = []

    # generate gt datas
    all_anchor_size = anchor_size # get_total_anchor_size(multi_level=True, name=name, version=version)
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+1+4+1+4]))
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            xmin, ymin, xmax, ymax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2 * w
            c_y = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1e-28 or box_h < 1e-28:
                # print('A dirty data !!!')
                continue    

            # compute the IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            #iou = compute_iou(anchor_boxes, gt_box)
            iou=bbox_overlaps_ciou(anchor_boxes,gt_box)
            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.
                # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other 
                # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
                # iou_ = iou * iou_mask
                
                # We get the index of the best IoU
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # get the corresponding stride
                            s = strides[s_indx]
                            # get the corresponding anchor box
                            p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                            # compute the gride cell location
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            # compute gt labels
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            tw = np.log(box_w / p_w)
                            th = np.log(box_h / p_h)
                            weight = 2.0 - (box_w / w) * (box_h / h)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
                        else:
                            # we ignore other anchor boxes even if their iou scores are higher than ignore thresh
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = -1.0
    """
    @Kissrabbit 知乎
    这些在代码里应该不难看出来的。这里，我一一解释一下1+1+4+1+4：

    1-即obj的学习目标，0或者1，有无中心点落在这个网格里；

    1-即cls的学习目标，由于我用的是cross-entropy，所以只需要提供类别的index即可，比如在voc上，取值范围是0-19(共20类)；

    4-即bbox相关的学习目标：【tx, ty, tw, th】;

    1-每个bbox的loss权重，取值范围【1-2】；

    4-即gt box的【xmin,ymin,xmax,ymax】，用来后续计算iou用的，实际上我没有用~
    """
    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
    
    return gt_tensor


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i)

#引入focal loss
class focal(nn.Module):
 def __init__(self):
     super(focal,self).__init__()

 def forward(*x):
        print(len(x))
        tar,act=x[1].cpu().data,x[2].cpu().data
        alpha = 1
        gamma = 2
        print(tar.size())
        print(act.size())
        focal_loss =  torch.pow(torch.abs(tar - act), gamma)
        print(focal_loss)
        return focal_loss


def loss(pred_conf, pred_cls, pred_txtytwth, label, num_classes, obj_loss_f='bce'):
    if obj_loss_f == 'bce':
        # In yolov3, we use bce as conf loss_f
        conf_loss_function = BCELoss(reduction='mean')
        #conf_loss_function = nn.CrossEntropyLoss(reduction='none')
        obj = 5.0
        noobj = 1.0
    elif obj_loss_f == 'mse':
        # In yolov2, we use mse as conf loss_f.
        conf_loss_function = MSELoss(reduction='mean')
        obj = 5.0
        noobj = 1.0
    # elif obj_loss_f=='focal':
    #     conf_loss_function=focal_loss()
    #类别损失函数
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    #x y的损失函数
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    #w h的损失函数
    twth_loss_function = nn.MSELoss(reduction='none')

    #预测的分数
    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    #预测的类别
    pred_cls = pred_cls.permute(0, 2, 1)

    #预测的x y
    txty_pred = pred_txtytwth[:, :, :2]
    #预测的 w h
    twth_pred = pred_txtytwth[:, :, 2:]
        
    gt_conf = label[:, :, 0].float()
    gt_obj = label[:, :, 1].float()
    gt_cls = label[:, :, 2].long()
    gt_txtytwth = label[:, :, 3:-1].float()
    gt_box_scale_weight = label[:, :, -1]
    gt_mask = (gt_box_scale_weight > 0.).float()

    """
    focal loss
    """
    #conf_focal = focal()
   # t=(gt_conf, pred_conf)
   # _=conf_focal(*t)

    #print(_.size())
    # objectness loss
    #conf_loss=conf_loss_function(pred_conf, gt_conf,gt_obj)
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    conf_losses = obj * pos_loss + noobj * neg_loss
    #print("conf_loss:",conf_loss.size())
    #focal

    
    # class loss
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask, 1))
    
    # box loss
    txty_loss = torch.mean(torch.sum(torch.sum(txty_loss_function(txty_pred, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_mask, 1))
    twth_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(twth_pred, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_mask, 1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_losses + cls_loss + txtytwth_loss

    return conf_losses, cls_loss, txtytwth_loss, total_loss

# IoU and its a series of variants
def IoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]

        Output: IoU  -> [iou_1, iou_2, ...],    size=[B, H*W]
    """

    return


def GIoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]

        Output: GIoU -> [giou_1, giou_2, ...],    size=[B, H*W]
    """

    return


def DIoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]

        Output: DIoU -> [diou_1, diou_2, ...],    size=[B, H*W]
    """

    return


def CIoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]

        Output: CIoU -> [ciou_1, ciou_2, ...],    size=[B, H*W]
    """

    return


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)