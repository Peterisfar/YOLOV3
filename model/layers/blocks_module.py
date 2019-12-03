import torch.nn as nn
from ..layers.conv_module import Convolutional


class Residual_block(nn.Module):
    def __init__(self, filters_in, filters_out, filters_medium):

        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in, filters_out=filters_medium, kernel_size=1, stride=1, pad=0,
                                     norm="bn", activate="leaky")
        self.__conv2 = Convolutional(filters_in=filters_medium, filters_out=filters_out, kernel_size=3, stride=1, pad=1,
                                     norm="bn", activate="leaky")

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r

        return out


class InvertedResidual_block(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_block, self).__init__()

        self.__stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.__stride == 1 and inp==oup

        if expand_ratio==1:
            self.__conv = nn.Sequential(
                Convolutional(filters_in=hidden_dim, filters_out=hidden_dim, kernel_size=3, stride=self.__stride,
                              pad=1, groups=hidden_dim, norm="bn", activate="relu6"),
                Convolutional(filters_in=hidden_dim, filters_out=oup, kernel_size=1, stride=1,
                              pad=0, norm="bn")
            )

        else:
            self.__conv = nn.Sequential(
                Convolutional(filters_in=inp, filters_out=hidden_dim, kernel_size=1, stride=1,
                              pad=0, norm="bn", activate="relu6"),
                Convolutional(filters_in=hidden_dim, filters_out=hidden_dim, kernel_size=3, stride=self.__stride,
                              pad=1, groups=hidden_dim, norm="bn", activate="relu6"),
                Convolutional(filters_in=hidden_dim, filters_out=oup, kernel_size=1, stride=1,
                              pad=0, norm="bn")
            )


    def forward(self, x):
        if self.use_res_connect:
            return x + self.__conv(x)
        else:
            return self.__conv(x)