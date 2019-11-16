import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
import numpy as np
import params as pms


class YoloV3Loss(object):
    def __init__(self,  focal_loss=False, iou_threshold_loss=0.5):
        self.__focal_loss = focal_loss
        self.__iou_threshold_loss = iou_threshold_loss

    def __call__(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        分层计算loss值。
        :param p: 预测偏移值。shape为[p0, p1, p2]共三个检测层，其中以p0为例，其shape为(bs,  grid, grid, anchors, tx+ty+tw+th+conf+cls_20)
        :param p_d: 解码后的预测值。shape为[pd0, pd1, pd2]，其中以pd0为例，其shape为(bs,  grid, grid, anchors, x+y+w+h+conf+cls_20)
        :param label_sbbox: small 检测层的分配标签, shape为[bs,  grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_mbbox: medium检测层的分配标签, shape为[bs,  grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_lbbox: large检测层的分配标签, shape为[bs,  grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param sbboxes: small检测层的bboxes, shape为[bs, 150, x+y+w+h+cls]
        :param mbboxes: medium检测层的bboxes, shape为[bs, 150, x+y+w+h+cls]
        :param lbboxes: large检测层的bboxes, shape为[bs, 150, x+y+w+h+cls]
        :return: loss为总损失值，loss_l[m, s]为每层的损失值
        """
        anchors = torch.tensor(pms.ANCHORS)
        strides = pms.STRIDES
        loss_s, loss_s_xywh, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_sbbox,
                                                               sbboxes, anchors[0], strides[0])
        loss_m, loss_m_xywh, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                               mbboxes, anchors[1], strides[1])
        loss_l, loss_l_xywh, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_lbbox,
                                                               lbboxes, anchors[2], strides[2])
        loss = (loss_l + loss_m + loss_s)/3
        loss_xywh = (loss_s_xywh + loss_m_xywh + loss_l_xywh) / 3
        loss_conf = (loss_s_conf + loss_m_conf + loss_l_conf) / 3
        loss_cls = (loss_s_cls + loss_s_cls + loss_l_cls) / 3

        return loss, loss_xywh, loss_conf, loss_cls


    def __cal_loss_per_layer(self, p, p_d, label, bboxes, anchors, stride):
        """
        计算每一层的损失。损失由三部分组成，(1)boxes的回归损失。计算预测的偏移量和标签的偏移量之间的损失。其中
        首先需要将标签的坐标转换成该层的相对于每个网格的偏移量以及长宽相对于每个anchor的比例系数。
        注意：损失的系数为2-w*h/(img_size**2),用于在不同尺度下对损失值大小影响不同的平衡。
        (2)置信度损失。包括前景和背景的置信度损失值。其中背景损失值需要注意的是在某特征点上标签为背景并且该点预测的anchor
        与该image所有bboxes的最大iou小于阈值时才计算该特征点的背景损失。
        (3)类别损失。类别损失为BCE，即每个类的二分类值。
        :param p: 没有进行解码的预测值，表示形式为(bs,  grid, grid, anchors, tx+ty+tw+th+conf+classes)
        :param p_d: p解码以后的结果。xywh均为相对于原图的尺度和位置，conf和cls均进行sigmoid。表示形式为
        [bs, grid, grid, anchors, x+y+w+h+conf+cls]
        :param label: lable的表示形式为(bs,  grid, grid, anchors, x+y+w+h+conf+cls) 其中xywh均为相对于原图的尺度和位置。
        :param bboxes: 该batch内分配给该层的所有bboxes，shape为(bs, 150, 5).
        :param anchors: 该检测层的ahchor尺度大小。格式为torch.tensor
        :param stride: 该层feature map相对于原图的尺度缩放量
        :return: 该检测层的所有batch平均损失。loss=loss_xywh + loss_conf + loss_cls。
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        MSE = nn.MSELoss(reduction="none")


        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        device = p.device

        p_dxdy = p[..., 0:2]
        p_dwdh = p[..., 2:4]
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., :4]  # 用于计算iou
        # p_d_conf = p_d[..., 4:5]   # 用于计算focal loss

        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 5:]

        # loss xywh
        ## label的坐标转换为tx,ty,tw,th
        y = torch.arange(0, grid).unsqueeze(1).repeat(1, grid)
        x = torch.arange(0, grid).unsqueeze(0).repeat(grid, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)


        label_txty = (1.0 * label_xywh[..., :2] / stride) - grid_xy
        label_twth = torch.log((1.0 * label_xywh[..., 2:] / stride) / anchors.to(device))
        label_twth = torch.where(torch.isinf(label_twth), torch.zeros_like(label_twth), label_twth)

        # bbox的尺度缩放权值
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)

        loss_xy = label_obj_mask * bbox_loss_scale * BCE(input=p_dxdy, target=label_txty)
        loss_wh = 0.5 * label_obj_mask * bbox_loss_scale * MSE(input=p_dwdh, target=label_twth)


        # loss confidence
        iou = tools.iou_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = label_obj_mask * BCE(input=p_conf, target=label_obj_mask) + \
                    label_noobj_mask * BCE(input=p_conf, target=label_obj_mask)

        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls)

        # loss = torch.cat([loss_xy, loss_wh, loss_conf, loss_cls], dim=-1)
        # loss = loss.sum([1,2,3,4], keepdim=True).mean()  # batch的平均损失
        # loss_xywh = torch.cat([loss_xy, loss_wh], dim=-1)

        loss_xywh = (torch.sum(loss_xy) + torch.sum(loss_wh)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size

        loss = loss_xywh + loss_conf + loss_cls

        return loss, loss_xywh, loss_conf, loss_cls


if __name__ == "__main__":
    from model import Darknet
    net = Darknet("cfg/yolov3-voc.cfg")

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3,  52, 52, 3,25)
    label_mbbox = torch.rand(3,  26, 26, 3, 25)
    label_lbbox = torch.rand(3, 13, 13, 3,25)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss()(p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)


    # TODO 每一层的anchor数量不一致，这里存在不平衡的情况