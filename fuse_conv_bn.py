# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 11:11  2022-06-07
import torch.nn as nn
import torch
import time
import torchvision
import numpy as np
import random

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class Conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.relu(self.conv(x))
    #
    # @torch.no_grad()
    # def fuse(self, x):
    #     fused = torch.nn.Conv2d(
    #         self.conv.in_channels,
    #         self.conv.out_channels,
    #         kernel_size=self.conv.kernel_size,
    #         stride=self.conv.stride,
    #         padding=self.conv.padding,
    #         bias=True
    #     )
    #
    #     # get conv layer weight and bias
    #     w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
    #     if self.conv.bias is not None:
    #         b_conv = self.conv.bias
    #     else:
    #         b_conv = torch.zeros(self.conv.weight.size(0))
    #
    #     # # get BN layer param
    #     r = self.bn.weight
    #     var = self.bn.running_var
    #     eps = self.bn.eps
    #     mean = self.bn.running_mean
    #
    #     # # cal new conv weight
    #     w_bn = torch.diag(r.div(torch.sqrt(var + eps)))
    #     new_conv_weight = torch.mm(w_bn, w_conv).view(fused.weight.size())
    #     # # copy new weight
    #     fused.weight.copy_(new_conv_weight)
    #
    #     # cal new conv bias
    #     b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
    #     b_bn = self.bn.bias - r.mul(mean).div(torch.sqrt(var + eps))
    #     new_conv_bias = b_conv + b_bn
    #
    #     fused.bias.copy_(new_conv_bias)
    #
    #     return fused.forward(x)


class Model(nn.Module):
    def __init__(self, fuse=False):
        super(Model, self).__init__()
        self.fuse = fuse
        self.conv1 = self.__makeLayer(2, 3, 64)
        self.conv2 = self.__makeLayer(3, 64, 128)

        self.conv3 = self.__makeLayer(3, 128, 256)
        self.conv4 = self.__makeLayer(3, 256, 512)
        # self.pool = nn.AvgPool2d(kernel_size=1, stride=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal(m.weight.data)
        #         # m.bias.data.fill_(0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_()
        #         m.bias.data.normal_()
        #         print(m.bias)

        if self.fuse:
            for m in self.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    # print("fuse conv and bn")
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward

    def __makeLayer(self, layerNum, inchannel, outchannel):
        layer = nn.Sequential()
        inF = inchannel
        outF = outchannel // 2
        for i in range(layerNum):
            # layer.append(Conv(inF, outF, fuse=self.fuse))
            layer.add_module("conv_{}".format(i), Conv(inF, outF))
            inF = outF
            outF = outchannel

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x0 = self.conv4(x)
        return x0


def test_time(net):
    a = torch.randn(2, 3, 224, 224)

    test_times = 20
    time_calculate = 0
    for _ in range(test_times):
        t1 = time.time()
        out = net(a)
        t2 = time.time()
        time_calculate += (t2 - t1)
    print("train mode mean times : ", out.shape, time_calculate / test_times)


@torch.no_grad()
def fuse(conv, bn, x):
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )

    # get conv layer weight and bias
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    # # get BN layer param
    r = bn.weight
    var = bn.running_var
    eps = bn.eps
    mean = bn.running_mean

    # # cal new conv weight
    w_bn = torch.diag(r.div(torch.sqrt(var + eps)))
    new_conv_weight = torch.mm(w_bn, w_conv).view(fused.weight.size())
    # # copy new weight
    fused.weight.copy_(new_conv_weight)

    # cal new conv bias
    b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
    b_bn = bn.bias - r.mul(mean).div(torch.sqrt(var + eps))
    new_conv_bias = b_conv + b_bn

    fused.bias.copy_(new_conv_bias)

    return fused.forward(x)


# if __name__ == '__main__':
#
#     resnet18 = torchvision.models.resnet18(pretrained=True)
#     # removing all learning variables, etc
#     resnet18.eval()
#     model = torch.nn.Sequential(
#         resnet18.conv1,
#         resnet18.bn1
#     )
#
#     x = torch.randn(2, 3, 224, 224).float()
#     f1 = model.forward(x)
#     fused3 = fuse(model[0], model[1], x)
#
#     # f3 = fused3.forward(x)
#
#     d3 = (f1 - fused3).sum().item()
#     print("error:", d3)

if __name__ == '__main__':

    # 官网实现
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/fusion.py
    # yolov5 fuse conv and bn layer
    # https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
    # def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
    #     LOGGER.info('Fusing layers... ')
    #     for m in self.model.modules():
    #         if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
    #             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #             delattr(m, 'bn')  # remove batchnorm
    #             m.forward = m.forward_fuse  # update forward

    # #
    # a = torch.randn(2, 3, 224, 224)
    # net1 = Model(fuse=False)
    # net1.eval()
    # with torch.no_grad():
    #     out1 = net1(a)
    # print("-"*20)
    # net2 = Model(fuse=True)
    # # print(net2)
    # net2.eval()
    # with torch.no_grad():
    #     out2 = net2(a)
    #
    # error = (out1 - out2).sum().item()
    # print(error)

    # 保存模型，并加载融合 conv + bn
    net1 = Model()
    torch.save(net1, "model.pt")

    model = torch.load("model.pt")
    a = torch.randn(2, 3, 448, 448)
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        out1 = model(a)
    t2 = time.time()
    print(t2 - t1)

    for m in model.modules():
        if isinstance(m, Conv) and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, 'bn')  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    t3 = time.time()
    with torch.no_grad():
        out2 = model(a)
    t4 = time.time()
    print(t4 - t3)
    error = (out1 - out2).sum().item()
    print(error)
