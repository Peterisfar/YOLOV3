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
                        focal_loss=False,
                        iou_reject=True,
                        iou_threshold_loss=0.5,
                        label_smoothing=False
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
        device = p[0].device
        lxy = torch.tensor([0]).float().to(device)
        lwh = torch.tensor([0]).float().to(device)
        lcls = torch.tensor([0]).float().to(device)
        lconf = torch.tensor([0]).float().to(device)
        txy, twh, tcls, indices = self.__build_targets(net, targets)

        # Define criteria
        MSE = nn.MSELoss()
        CE = nn.CrossEntropyLoss()
        BCE = nn.BCEWithLogitsLoss()

        # Compute losses
        bs = p[0].shape[0]  # batch size
        k = 10.39 * bs  # loss gain
        for i, pi0 in enumerate(p):  # layer i predictions, i
            b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
            tconf = torch.zeros_like(pi0[..., 0])  # conf

            # Compute losses
            if len(b):  # number of targets
                pi = pi0[b, a, gj, gi]  # predictions closest to anchors
                tconf[b, a, gj, gi] = 1  # conf

                lxy += (k * 0.13) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
                lwh += (k * 0.01) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss
                lcls += (k * 0.01) * CE(pi[..., 5:], tcls[i])  # class_conf loss

            lconf += (k * 0.84) * BCE(pi0[..., 4], tconf)  # obj_conf loss
        loss = lxy + lwh + lconf + lcls

        return loss

    def __build_targets(self, net, targets):
        """
        :param net:
        :param targets: targets shape : [bs*N, 7] ---- [x, y, w, h, cls, mixup_weight, bs_id]
        :return:shape : [bs, nA, grid, grid, 5+cls+1+2] ------[tx,ty,tw,th,conf,mixup_weight,cls...,gw,gh]
        """
        # 根据每个检测层分配label
        # targets = [image, class, x, y, w, h]
        iou_thres = self.IOU_THRESHOLD_LOSS

        nt = len(targets)
        txy, twh, tcls, indices = [], [], [], []
        for i in net.yolo_layers_num:
            layer = net.module_list[i][0]

            # iou of targets-anchors
            t, a = targets, []
            gwh = targets[:, 2:4] * layer.nG
            if nt:
                iou = [tools.wh_iou(x, gwh) for x in layer.anchor_vec]
                iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor

                # reject below threshold ious (OPTIONAL, increases P, lowers R)
                reject = True
                if reject:
                    j = iou > iou_thres
                    t, a, gwh = targets[j], a[j], gwh[j]

            # Indices
            b = t[:, -1].long()
            c = t[:, 4].long()
            gxy = t[:, :2] * layer.nG
            gi, gj = gxy.long().t()  # grid_i, grid_j
            indices.append((b, a, gj, gi))

            # XY coordinates
            txy.append(gxy - gxy.floor())

            # Width and height
            twh.append(torch.log(gwh / layer.anchor_vec[a]))  # wh yolo method

            # Class
            tcls.append(c)
            if c.shape[0]:
                assert c.max() <= layer.nC, 'Target classes exceed model classes'

        return txy, twh, tcls, indices


# if __name__ == "__main__":
#     from model import Darknet
#     # test loss is ok!
#     net = Darknet("cfg/yolov3-voc.cfg")
#
#     p = net(torch.rand(3, 3, 416, 416))
#     target = torch.rand(8, 7)
#     loss = YoloV3Loss()(net, p, target)
#     print(loss)