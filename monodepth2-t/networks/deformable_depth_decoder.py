from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class Deformable_DepthDecoder(nn.Module):
    def __init__(self):
        super(Deformable_DepthDecoder, self).__init__()
        self.upsample_mode = 'nearest'

        # decoder
        self.convs = OrderedDict()

        self.convs[("upconv", 1, 0)] = ConvBlock(512, 256)
        self.convs[("upconv", 1, 1)] = ConvBlock(512, 256)
        self.convs[("dispconv", 4)] = Conv3x3(256, 1)

        self.convs[("upconv", 2, 0)] = DepModule_4(256, 128, 448, 224, 448, 128, 128)
        self.convs[("dispconv", 3)] = Conv3x3(128, 1)

        self.convs[("upconv", 3, 0)] = DepModule_4(128, 64, 224, 112, 224, 64, 64)
        self.convs[("dispconv", 2)] = Conv3x3(64, 1)

        self.convs[("upconv", 4, 0)] = DepModule_4(64, 32, 160, 80, 160, 32, 64)
        self.convs[("dispconv", 1)] = Conv3x3(32, 1)

        self.convs[("upconv", 5, 0)] = ConvBlock(32, 16)
        self.convs[("upconv", 5, 1)] = ConvBlock(32, 16)
        self.convs[("upconv", 5, 2)] = ConvBlock_1(32, 16)
        self.convs[("aug", 5, 0)] = ops.DeformConv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.convs[("aug", 5, 1)] = ops.DeformConv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.convs[("ffs", 5, 0)] = SAF2(32)
        self.convs[("ffs", 5, 1)] = SAF2(32)
        self.convs[("dispconv", 0)] = Conv3x3(16, 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, flag, depth_map, para_k):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        x = self.convs[("upconv", 1, 0)](x)
        x = [upsample(x)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        self.outputs[("disp", 4)] = self.sigmoid(self.convs[("dispconv", 4)](x))
        _, depth_4 = disp_to_depth(self.outputs[("disp", 4)], 0.1, 100)

        if flag:
            offset_256, scale_256 = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k)
            offset_128, scale_128 = depth_to_scale(depth_map, input_features[2].shape[2], input_features[2].shape[3], para_k)
        else:
            offset_256, scale_256 = depth_to_scale(depth_4, x.shape[2], x.shape[3], para_k)
            offset_128, scale_128 = depth_to_scale(depth_4, input_features[2].shape[2], input_features[2].shape[3], para_k)
        x = self.convs[("upconv", 2, 0)](x, offset_256, input_features[2], offset_128, scale_256, scale_128)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](x))
        _, depth_3 = disp_to_depth(self.outputs[("disp", 3)], 0.1, 100)

        if flag:
            offset_64, scale_64 = depth_to_scale(depth_map, input_features[1].shape[2], input_features[1].shape[3], para_k)
        else:
            offset_128, scale_128 = depth_to_scale(depth_3, x.shape[2], x.shape[3], para_k)
            offset_64, scale_64 = depth_to_scale(depth_3, input_features[1].shape[2], input_features[1].shape[3], para_k)
        x = self.convs[("upconv", 3, 0)](x, offset_128, input_features[1], offset_64, scale_128, scale_64)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](x))
        _, depth_2 = disp_to_depth(self.outputs[("disp", 2)], 0.1, 100)

        if flag:
            offset_64_1, scale_64_1 = depth_to_scale(depth_map, input_features[0].shape[2], input_features[0].shape[3], para_k)
        else:
            offset_64, scale_64 = depth_to_scale(depth_2, x.shape[2], x.shape[3], para_k)
            offset_64_1, scale_64_1 = depth_to_scale(depth_2, input_features[0].shape[2], input_features[0].shape[3], para_k)
        x = self.convs[("upconv", 4, 0)](x, offset_64, input_features[0], offset_64_1, scale_64, scale_64_1)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](x))
        _, depth_1 = disp_to_depth(self.outputs[("disp", 1)], 0.1, 100)

        if flag:
            offset, scale_small = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k)
        else:
            offset, scale_small = depth_to_scale(depth_1, x.shape[2], x.shape[3], para_k)
        aug_feature_fir = self.convs[("aug", 5, 0)](x, offset)
        x = self.convs[("upconv", 5, 0)](x)
        x = self.convs[("ffs", 5, 0)](x, aug_feature_fir, scale_small)
        x = upsample(x)
        if flag:
            offset, scale_large = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k)
        else:
            offset, scale_large = depth_to_scale(depth_1, x.shape[2], x.shape[3], para_k)
        aug_feature_sec = self.convs[("aug", 5, 1)](x, offset)
        x = self.convs[("upconv", 5, 1)](x)
        x = self.convs[("ffs", 5, 1)](x, aug_feature_sec, scale_large)
        x = self.convs[("upconv", 5, 2)](x)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](x))

        return self.outputs
