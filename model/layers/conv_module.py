import torch
import torch.nn as nn
import torch.nn.functional as F
from .activate import *


norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "mish": Mish,
    "relu6": nn.ReLU6}


class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, norm=None, activate=None):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.activate = activate

        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm, groups=groups)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)


    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x



class Separable_Conv(nn.Module):
    def __init__(self, filters_in, filters_out, stride):
        super(Separable_Conv, self).__init__()

        self.__dw = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3, stride=stride,
                                  pad=1, groups=filters_in, norm="bn", activate="relu6")

        self.__pw = Convolutional(filters_in=filters_in, filters_out=filters_out, kernel_size=1, stride=1,
                                  pad=0, norm="bn", activate="relu6")

    def forward(self, x):
        return self.__pw(self.__dw(x))