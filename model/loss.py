import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
import params as pms


class YoloV3Loss(object):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        self.__iou_threshold_loss = iou_threshold_loss
        self.__anchors = anchors
        self.__strides = strides


    def __call__(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        anchors = torch.tensor(self.__anchors)
        strides = self.__strides

        loss_s, loss_s_xywh, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_sbbox,
                                                               sbboxes, anchors[0], strides[0])
        loss_m, loss_m_xywh, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                               mbboxes, anchors[1], strides[1])
        loss_l, loss_l_xywh, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_lbbox,
                                                               lbboxes, anchors[2], strides[2])

        loss = (loss_l + loss_m + loss_s) / 3
        loss_xywh = (loss_s_xywh + loss_m_xywh + loss_l_xywh) / 3
        loss_conf = (loss_s_conf + loss_m_conf + loss_l_conf) / 3
        loss_cls = (loss_s_cls + loss_s_cls + loss_l_cls) / 3

        return loss, loss_xywh, loss_conf, loss_cls


    def __cal_loss_per_layer(self, p, p_d, label, bboxes, anchors, stride):
        """
        (1)The loss of regression of boxes.
          Calculate the loss between the predicted offset and the offset of the label.
        The coordinates of the label need to be converted to the offset of the layer
        relative to each grid and the scale factor of the length and width relative
        to each anchor.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param anchors: The ahchors of the detection layer. The format is torch.tensor
        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_xywh, loss_conf, loss_cls) of all batches of this detection layer.
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

        p_d_xywh = p_d[..., :4]

        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 5:]

        # loss xywh
        ## The coordinates of the label are converted to tx, ty, tw, th
        y = torch.arange(0, grid).unsqueeze(1).repeat(1, grid)
        x = torch.arange(0, grid).unsqueeze(0).repeat(grid, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)
        label_txty = (1.0 * label_xywh[..., :2] / stride) - grid_xy
        label_twth = torch.log((1.0 * label_xywh[..., 2:] / stride) / anchors.to(device))
        label_twth = torch.where(torch.isinf(label_twth), torch.zeros_like(label_twth), label_twth)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)

        loss_xy = label_obj_mask * bbox_loss_scale * BCE(input=p_dxdy, target=label_txty)
        loss_wh = 0.5 * label_obj_mask * bbox_loss_scale * MSE(input=p_dwdh, target=label_twth)


        # loss confidence
        iou = tools.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = label_obj_mask * BCE(input=p_conf, target=label_obj_mask) + \
                    label_noobj_mask * BCE(input=p_conf, target=label_obj_mask)

        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls)


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

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(pms.MODEL["ANCHORS"], pms.MODEL["STRIDES"])(p, p_d, label_sbbox,
                                    label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)
