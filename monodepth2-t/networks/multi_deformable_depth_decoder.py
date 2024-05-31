from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class Multi_Deformable_DepthDecoder(nn.Module):
    def __init__(self):
        super(Multi_Deformable_DepthDecoder, self).__init__()
        self.upsample_mode = 'nearest'

        # decoder
        self.convs = OrderedDict()

        self.convs[("upconv", 1, 0)] = ConvBlock(512, 256)
        self.convs[("upconv", 1, 1)] = ConvBlock(512, 256)
        self.convs[("dispconv", 4)] = Conv3x3(256, 1)

        # self.convs[("upconv", 2, 0)] = DepModule_1(256, 128, 448, 224, 448, 128, 128)
        # self.convs[("upconv", 2, 0)] = DepModule_2(256, 64, 288, 64, 128)
        # self.convs[("upconv", 2, 0)] = DepModule_3(256, 128, 320, 128, 128)
        self.convs[("upconv", 2, 0)] = Multi_DepModule_4(256, 128, 448, 224, 448, 128, 128)
        # self.convs[("upconv", 2, 0)] = DepModule_5(256, 128, 352, 128, 192, 128, 128)
        # self.convs[("upconv", 2, 0)] = DepModule_6(256, 64, 288, 64, 128)
        # self.convs[("upconv", 2, 0)] = DepModule_7(256, 64, 288, 64, 128)
        self.convs[("dispconv", 3)] = Conv3x3(128, 1)

        # self.convs[("upconv", 3, 0)] = DepModule_1(128, 64, 224, 112, 224, 64, 64)
        # self.convs[("upconv", 3, 0)] = DepModule_2(128, 32, 144, 32, 64)
        # self.convs[("upconv", 3, 0)] = DepModule_3(128, 64, 160, 64, 64)
        self.convs[("upconv", 3, 0)] = Multi_DepModule_4(128, 64, 224, 112, 224, 64, 64)
        # self.convs[("upconv", 3, 0)] = DepModule_5(128, 64, 176, 64, 96, 64, 64)
        # self.convs[("upconv", 3, 0)] = DepModule_6(128, 32, 144, 32, 64)
        # self.convs[("upconv", 3, 0)] = DepModule_7(128, 32, 144, 32, 64)
        self.convs[("dispconv", 2)] = Conv3x3(64, 1)

        # self.convs[("upconv", 4, 0)] = DepModule_1(64, 32, 160, 80, 160, 32, 64)
        # self.convs[("upconv", 4, 0)] = DepModule_2(64, 16, 112, 16, 64)
        # self.convs[("upconv", 4, 0)] = DepModule_3(64, 32, 128, 32, 64)
        self.convs[("upconv", 4, 0)] = Multi_DepModule_4(64, 32, 160, 80, 160, 32, 64)
        # self.convs[("upconv", 4, 0)] = DepModule_5(64, 32, 128, 64, 80, 32, 64)
        # self.convs[("upconv", 4, 0)] = DepModule_6(64, 16, 112, 16, 64)
        # self.convs[("upconv", 4, 0)] = DepModule_7(64, 16, 112, 16, 64)
        self.convs[("dispconv", 1)] = Conv3x3(32, 1)

        self.convs[("upconv", 5, 0)] = ConvBlock(32, 16)
        self.convs[("upconv", 5, 1)] = ConvBlock(32, 16)
        self.convs[("upconv", 5, 2)] = ConvBlock_1(32, 16)
        self.convs[("aug", 5, 0)] = Multi_Deformable(32, 16)
        self.convs[("aug", 5, 1)] = Multi_Deformable(32, 16)
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
            offset_256_m, offset_256_f, offset_256_c = multi_depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k)
            offset_128_m, offset_128_f, offset_128_c = multi_depth_to_scale(depth_map, input_features[2].shape[2], input_features[2].shape[3], para_k)
        else:
            offset_256_m, offset_256_f, offset_256_c = multi_depth_to_scale(depth_4, x.shape[2], x.shape[3], para_k)
            offset_128_m, offset_128_f, offset_128_c = multi_depth_to_scale(depth_4, input_features[2].shape[2], input_features[2].shape[3], para_k)
        x = self.convs[("upconv", 2, 0)](x, offset_256_m, offset_256_f, offset_256_c,
                                         input_features[2], offset_128_m, offset_128_f, offset_128_c)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](x))
        _, depth_3 = disp_to_depth(self.outputs[("disp", 3)], 0.1, 100)

        if flag:
            offset_64_m, offset_64_f, offset_64_c = multi_depth_to_scale(depth_map, input_features[1].shape[2], input_features[1].shape[3], para_k)
        else:
            offset_128_m, offset_128_f, offset_128_c = multi_depth_to_scale(depth_3, x.shape[2], x.shape[3], para_k)
            offset_64_m, offset_64_f, offset_64_c = multi_depth_to_scale(depth_3, input_features[1].shape[2], input_features[1].shape[3], para_k)
        x = self.convs[("upconv", 3, 0)](x, offset_128_m, offset_128_f, offset_128_c,
                                         input_features[1], offset_64_m, offset_64_f, offset_64_c)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](x))
        _, depth_2 = disp_to_depth(self.outputs[("disp", 2)], 0.1, 100)

        if flag:
            offset_64_1_m, offset_64_1_f, offset_64_1_c = multi_depth_to_scale(depth_map, input_features[0].shape[2], input_features[0].shape[3], para_k)
        else:
            offset_64_m, offset_64_f, offset_64_c = multi_depth_to_scale(depth_2, x.shape[2], x.shape[3], para_k)
            offset_64_1_m, offset_64_1_f, offset_64_1_c = multi_depth_to_scale(depth_2, input_features[0].shape[2], input_features[0].shape[3], para_k)
        x = self.convs[("upconv", 4, 0)](x, offset_64_m, offset_64_f, offset_64_c,
                                         input_features[0], offset_64_1_m, offset_64_1_f, offset_64_1_c)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](x))
        _, depth_1 = disp_to_depth(self.outputs[("disp", 1)], 0.1, 100)

        if flag:
            offset_m, offset_f, offset_c = multi_depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k)
        else:
            offset_m, offset_f, offset_c = multi_depth_to_scale(depth_1, x.shape[2], x.shape[3], para_k)
        aug_feature_fir = self.convs[("aug", 5, 0)](x, offset_m, offset_f, offset_c)
        x = self.convs[("upconv", 5, 0)](x)
        x = torch.cat([x, aug_feature_fir], 1)
        x = upsample(x)
        if flag:
            offset_m, offset_f, offset_c = multi_depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k)
        else:
            offset_m, offset_f, offset_c = multi_depth_to_scale(depth_1, x.shape[2], x.shape[3], para_k)
        aug_feature_sec = self.convs[("aug", 5, 1)](x, offset_m, offset_f, offset_c)
        x = self.convs[("upconv", 5, 1)](x)
        x = torch.cat([x, aug_feature_sec], 1)
        x = self.convs[("upconv", 5, 2)](x)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](x))

        return self.outputs
