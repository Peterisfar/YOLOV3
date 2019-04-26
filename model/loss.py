import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
from utils import data_augment as dataAug
import os


class YoloV3Loss(object):
    def __init__(self, coord_scale=1.0,
                        obj_scale=1.0,
                        noobj_scale=1.0,
                        class_scale=1.0,
                        focal_loss=True,
                        iou_reject=True,
                        iou_threshold_loss=0.5,
                        label_smoothing=True
                        ):
        self.coord_scale = coord_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.class_scale = class_scale
        self.FOCAL_LOSS = focal_loss
        self.IOU_THRESHOLD_REJECT = iou_reject
        self.IOU_THRESHOLD_LOSS=iou_threshold_loss
        self.LABEL_SMOOTHING = label_smoothing

    def __call__(self, net, p, targets):
        """
        :param net: 网络
        :param p: shape : 3 * [bs, anchors, grid, grid, 5+cls]  -------5+cls=[tx, ty, tw, th, conf, cls]
        :param targets: [bs*N, 6] ----6->[x,y,w,h,cls,mixup_weight]
        :return: loss
        """
        # adjust targets shape : 3 * [bs, nA, grid, grid, 5+1+cls+2] ------
        # 其中 5+1+cls+2 = [tx,ty,tw,th,conf,mixup_weight, cls..., gw, gh]
        bath_size = p[0].shape[0]
        targets = self.__build_targets(net, targets, p[0].shape[0])
        device = p[0].device
        MSE = nn.MSELoss(reduce=False, reduction=None)
        BCE = nn.BCEWithLogitsLoss(reduce=False, reduction=None)
        loss_bboxes = torch.tensor([0]).float().to(device)
        loss_cls = torch.tensor([0]).float().to(device)
        loss_conf = torch.tensor([0]).float().to(device)

        for i, pi in enumerate(p):
            mask_obj = targets[i][..., 4] == 1.0
            mask_noobj = targets[i][..., 4] == 0.0
            targets_obj = targets[i][mask_obj]
            targets_noobj = targets[i][mask_noobj]

            if targets_obj.shape[0]:
                # bboxes loss
                bbox_loss_scale = 2.0 - 1.0 * targets_obj[..., -2] * targets_obj[..., -1]
                # 计算tx ty tw th 损失并乘以mixup_weight
                loss_bboxes_base = (MSE(torch.sigmoid(pi[mask_obj][..., :2]), targets_obj[..., :2]) + \
                                    0.5 * MSE(pi[mask_obj][..., 2:4], targets_obj[..., 2:4])) * \
                                   targets_obj[..., 5].view(-1,1)  # mixup_weight
                loss_bboxes += self.coord_scale * torch.sum(bbox_loss_scale.view(-1, 1) * loss_bboxes_base)

                # class loss
                loss_cls += self.class_scale * torch.sum(BCE(pi[mask_obj][..., 5:], targets_obj[..., 6:-2]) *
                                                    targets_obj[..., 5].view(-1,1))

            # confidence loss
            conf_focal = self.__focal_loss(targets[i][..., 4], pi[..., 4]) if self.FOCAL_LOSS else \
                        torch.ones_like(pi[..., 4]) # focal_loss False 时使初始值为1.0
            loss_conf_obj = self.obj_scale * conf_focal[mask_obj] * \
                            BCE(pi[mask_obj][..., 4], targets_obj[..., 4]) * \
                            targets_obj[..., 5]
            loss_conf_noobj = self.noobj_scale * conf_focal[mask_noobj] * \
                              BCE(pi[mask_noobj][..., 4], targets_noobj[..., 4])
            loss_conf += torch.sum(torch.cat((loss_conf_obj, loss_conf_noobj)))

        loss = (loss_bboxes + loss_cls + loss_conf) / bath_size

        return loss

    def __focal_loss(self, target, p, alpha=1.0, gamma=2.0):
        return alpha * torch.pow(torch.abs(target - p), gamma)

    def __build_targets(self, net, targets, bs):
        """
        :param net:
        :param targets: targets shape : [bs*N, 7] ---- [x, y, w, h, cls, mixup_weight, bs_id]
        :return:shape : [bs, nA, grid, grid, 5+cls+1+2] ------[tx,ty,tw,th,conf,mixup_weight,cls...,gw,gh]
        """
        # 根据每个检测层分配label
        bboxes_g = []
        for i in net.yolo_layers_num:
            yolo_layer = net.module_list[i][0]
            bbox_g = torch.zeros(bs,
                                 yolo_layer.nA,
                                 int(yolo_layer.nG.item()),
                                 int(yolo_layer.nG.item()),
                                 6 + yolo_layer.nC + 2).to(targets.device)
            bbox_g[..., 5] = 1.0  # mixup_weight 初始值

            # 根据wh_iou指定GT由哪个anchor负责， 选择iou最大为基准
            gwh = targets[:, 2:4] * yolo_layer.nG  # gt 尺寸与feature map 一致
            # anchor尺寸为feature map 一致；x shape : [2] gwh [bs*N, 2] (广播)
            iou = [tools.wh_iou(x, gwh) for x in yolo_layer.anchor_vec]
            iou, a = torch.stack(iou, -1).max(-1)

            # iou threshold  选出IOU符合条件的anchor
            if self.IOU_THRESHOLD_REJECT:
                mask = iou > self.IOU_THRESHOLD_LOSS
                t, a, gwh = targets[mask], a[mask], gwh[mask]
            else:
                t = targets

            if len(t):
                bn = t[:, -1].long()
                gxy = t[:, :2] * yolo_layer.nG
                grid_x, grid_y = gxy.long().t()

                txy = gxy - gxy.floor()  # GT相对与所在网格左上角的坐标偏移量
                twh = torch.log(gwh / yolo_layer.anchor_vec[a])  # GT相对与所负责anchor的尺度缩放量
                mixup_weight = t[:, -2].view(-1,1).contiguous()
                conf = torch.ones(len(bn), 1).to(targets.device)

                cls = torch.zeros(len(bn), yolo_layer.nC).to(targets.device)
                c = t[:, -3].long()
                for i in range(len(c)):
                    cls[i, c[i]] = 1.0
                # label smoothing
                if self.LABEL_SMOOTHING:
                    cls = dataAug.LabelSmooth()(cls, yolo_layer.nC)

                # shape : [bs, nA, grid, grid, 5+cls+1+2] ------[tx,ty,tw,th,conf,mixup_weight, cls..., gw, gh]
                bbox_g[bn, a, grid_x, grid_y] = torch.cat((txy, twh, conf, mixup_weight, cls, gwh / yolo_layer.nG), -1)
                bboxes_g.append(bbox_g)
            else:
                bboxes_g.append(bbox_g)

        return bboxes_g


# if __name__ == "__main__":
#     from model import Darknet
#     # test loss is ok!
#     net = Darknet("cfg/yolov3-voc.cfg")
#
#     p = net(torch.rand(3, 3, 416, 416))
#     target = torch.rand(8, 7)
#     loss = YoloV3Loss()(net, p, target)
#     print(loss)