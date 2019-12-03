import sys
sys.path.append("..")

# AbsolutePath = os.path.abspath(__file__)           #将相对路径转换成绝对路径
# SuperiorCatalogue = os.path.dirname(AbsolutePath)   #相对路径的上级路径
# BaseDir = os.path.dirname(SuperiorCatalogue)        #在“SuperiorCatalogue”的基础上在脱掉一层路径，得到我们想要的路径。
# sys.path.insert(0,BaseDir)                          #将我们取出来的路径加入

import torch.nn as nn
import torch
from model.backbones.mobilenetv2 import MobilenetV2
from model.necks.yolo_fpn import FPN_YOLOV3
from model.head.yolo_head import Yolo_head
import config.yolov3_config_voc as cfg
from utils.tools import *


class Yolov3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(Yolov3, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])
        self.__fpn = FPN_YOLOV3(fileters_in=[1280, 96, 32],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        # small
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])


    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)


if __name__ == '__main__':
    net = Yolov3("../weight/mobilenetv2_1.0-0c6065bc.pth")
    print(net)

    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img)

    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
