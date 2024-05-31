from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class NewDepthDecoder(nn.Module):
    def __init__(self):
        super(NewDepthDecoder, self).__init__()
        self.upsample_mode = 'nearest'

        # decoder
        self.convs = OrderedDict()

        self.convs[("upconv", 1, 0)] = ConvBlock(512, 256)
        self.convs[("upconv", 1, 1)] = ConvBlock(512, 256)
        self.convs[("dispconv", 4)] = Conv3x3(256, 1)

        self.convs[("upconv", 2, 0)] = DepModule(256, 128, 448, 224, 448, 128, 128)
        self.convs[("dispconv", 3)] = Conv3x3(128, 1)

        self.convs[("upconv", 3, 0)] = DepModule(128, 64, 224, 112, 224, 64, 64)
        self.convs[("dispconv", 2)] = Conv3x3(64, 1)

        self.convs[("upconv", 4, 0)] = DepModule(64, 32, 160, 80, 160, 32, 64)
        self.convs[("dispconv", 1)] = Conv3x3(32, 1)

        self.convs[("upconv", 5, 0)] = ConvBlock(32, 16)
        self.convs[("upconv", 5, 1)] = ConvBlock(32, 16)
        self.convs[("upconv", 5, 2)] = ConvBlock_1(32, 16)
        self.convs[("aug", 5, 0)] = Depth_Scaler(32, 16)
        self.convs[("aug", 5, 1)] = Depth_Scaler(32, 16)
        self.convs[("dispconv", 0)] = Conv3x3(16, 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, flag, depth_map, para_k):
        self.outputs = {}

        x = input_features[-1]
        x = self.convs[("upconv", 1, 0)](x)
        x = [upsample(x)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        self.outputs[("disp", 4)] = self.sigmoid(self.convs[("dispconv", 4)](x))
        _, depth_4 = disp_to_depth(self.outputs[("disp", 4)], 0.1, 100)

        if flag:
            x = self.convs[("upconv", 2, 0)](x, input_features[2], depth_map, para_k)
        else:
            x = self.convs[("upconv", 2, 0)](x, input_features[2], depth_4, para_k)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](x))
        _, depth_3 = disp_to_depth(self.outputs[("disp", 3)], 0.1, 100)

        if flag:
            x = self.convs[("upconv", 3, 0)](x, input_features[1], depth_map, para_k)
        else:
            x = self.convs[("upconv", 3, 0)](x, input_features[1], depth_3, para_k)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](x))
        _, depth_2 = disp_to_depth(self.outputs[("disp", 2)], 0.1, 100)

        if flag:
            x = self.convs[("upconv", 4, 0)](x, input_features[0], depth_map, para_k)
        else:
            x = self.convs[("upconv", 4, 0)](x, input_features[0], depth_2, para_k)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](x))
        _, depth_1 = disp_to_depth(self.outputs[("disp", 1)], 0.1, 100)

        if flag:
            aug_feature_fir = self.convs[("aug", 5, 0)](x, depth_map, para_k)
        else:
            aug_feature_fir = self.convs[("aug", 5, 0)](x, depth_1, para_k)
        x = self.convs[("upconv", 5, 0)](x)
        x = torch.cat([x, aug_feature_fir], 1)
        x = upsample(x)
        if flag:
            aug_feature_sec = self.convs[("aug", 5, 1)](x, depth_map, para_k)
        else:
            aug_feature_sec = self.convs[("aug", 5, 1)](x, depth_1, para_k)
        x = self.convs[("upconv", 5, 1)](x)
        x = torch.cat([x, aug_feature_sec], 1)
        x = self.convs[("upconv", 5, 2)](x)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](x))

        return self.outputs


