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
        self.convs[("attentionConv", 1, 0)] = fSEModule(1280)
        self.convs[("upconv", 1, 1)] = ConvBlock_1(1280, 1024)
        self.convs[("upconv", 1, 2)] = ConvBlock_1(1024, 512)
        self.convs[("upconv", 1, 3)] = ConvBlock(512, 256)
        self.convs[("attentionConv", 1, 1)] = fSEModule(768)
        self.convs[("upconv", 1, 4)] = ConvBlock_1(768, 512)
        self.convs[("upconv", 1, 5)] = ConvBlock_1(512, 256)
        self.convs[("dispconv", 4)] = Conv3x3(256, 1)

        self.convs[("upconv", 2, 0)] = ConvBlock(256, 128)
        self.convs[("attentionConv", 2, 0)] = fSEModule(640)
        self.convs[("upconv", 2, 1)] = ConvBlock_1(640, 512)
        self.convs[("upconv", 2, 2)] = ConvBlock_1(512, 256)
        self.convs[("upconv", 2, 3)] = ConvBlock(256, 128)
        self.convs[("attentionConv", 2, 1)] = fSEModule(384)
        self.convs[("upconv", 2, 4)] = ConvBlock_1(384, 256)
        self.convs[("upconv", 2, 5)] = ConvBlock_1(256, 128)
        self.convs[("dispconv", 3)] = Conv3x3(128, 1)

        self.convs[("upconv", 3, 0)] = ConvBlock(128, 64)
        self.convs[("attentionConv", 3, 0)] = fSEModule(320)
        self.convs[("upconv", 3, 1)] = ConvBlock_1(320, 256)
        self.convs[("upconv", 3, 2)] = ConvBlock_1(256, 128)
        self.convs[("upconv", 3, 3)] = ConvBlock(128, 64)
        self.convs[("attentionConv", 3, 1)] = fSEModule(192)
        self.convs[("upconv", 3, 4)] = ConvBlock_1(192, 128)
        self.convs[("upconv", 3, 5)] = ConvBlock_1(128, 64)
        self.convs[("dispconv", 2)] = Conv3x3(64, 1)

        self.convs[("upconv", 4, 0)] = ConvBlock(64, 32)
        self.convs[("attentionConv", 4, 0)] = fSEModule(224)
        self.convs[("upconv", 4, 1)] = ConvBlock_1(224, 128)
        self.convs[("upconv", 4, 2)] = ConvBlock_1(128, 64)
        self.convs[("upconv", 4, 3)] = ConvBlock(64, 32)
        self.convs[("attentionConv", 4, 1)] = fSEModule(96)
        self.convs[("upconv", 4, 4)] = ConvBlock_1(96, 64)
        self.convs[("upconv", 4, 5)] = ConvBlock_1(64, 32)
        self.convs[("dispconv", 1)] = Conv3x3(32, 1)

        self.convs[("upconv", 5, 0)] = ConvBlock(32, 16)
        self.convs[("attentionConv", 5)] = fSEModule(48)
        self.convs[("upconv", 5, 1)] = ConvBlock_1(48, 32)
        self.convs[("upconv", 5, 2)] = ConvBlock_1(32, 16)
        self.convs[("dispconv", 0)] = Conv3x3(16, 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, flag, depth_map, model_depth_512, model_depth_256, model_depth_128, model_depth_64, model_depth_32, para_k, para_b):
        self.outputs = {}
        model_512 = model_depth_512
        model_256 = model_depth_256
        model_128 = model_depth_128
        model_64 = model_depth_64
        model_32 = model_depth_32

        """
            第0次编码和解码特征融合增强
        """
        x = input_features[-1]
        # 512->512  aug_enc_feature_512_1
        offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
        aug_enc_feature_512_1 = model_512(x, offset)
        # 512->256 x
        x = self.convs[("upconv", 1, 0)](x)
        # 256+512->768 x
        x = torch.cat([x, aug_enc_feature_512_1], 1)
        x = [upsample(x)]
        # 256->256 aug_dec_feature_256
        offset = depth_to_scale(depth_map, input_features[3].shape[2], input_features[3].shape[3], para_k, para_b)
        aug_dec_feature_256 = model_256(input_features[3], offset)
        # 256+256->512 hybrid_feature_512
        hybrid_feature_512 = torch.cat([aug_dec_feature_256, input_features[3]], 1)
        # 768+512->1280 x
        x += [hybrid_feature_512]
        x = torch.cat(x, 1)
        x = self.convs[("attentionConv", 1, 0)](x)
        # 1280->1024->512 x
        x = self.convs[("upconv", 1, 1)](x)
        x = self.convs[("upconv", 1, 2)](x)
        # 512->512  aug_enc_feature_512_2
        offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
        aug_enc_feature_512_2 = model_512(x, offset)
        # 512->256 x
        x = self.convs[("upconv", 1, 3)](x)
        # 256+512->768 x
        x = torch.cat([x, aug_enc_feature_512_2], 1)
        x = self.convs[("attentionConv", 1, 1)](x)
        # 768->512->256 x
        x = self.convs[("upconv", 1, 4)](x)
        x = self.convs[("upconv", 1, 5)](x)
        self.outputs[("disp", 4)] = self.sigmoid(self.convs[("dispconv", 4)](x))
        _, depth_4 = disp_to_depth(self.outputs[("disp", 4)], 0.1, 100)

        """
            第一次编码和解码特征融合增强
        """
        # 256->256  aug_enc_feature_256_1
        if flag:
            offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_256_1 = model_256(x, offset)
        else:
            offset = depth_to_scale(depth_4, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_256_1 = model_256(x, offset)
        # 256->128 x
        x = self.convs[("upconv", 2, 0)](x)
        # 256+128->384 x
        x = torch.cat([x, aug_enc_feature_256_1], 1)
        x = [upsample(x)]
        # 128->128 aug_dec_feature_128
        if flag:
            offset = depth_to_scale(depth_map, input_features[2].shape[2], input_features[2].shape[3], para_k, para_b)
            aug_dec_feature_128 = model_128(input_features[2], offset)
        else:
            offset = depth_to_scale(depth_4, input_features[2].shape[2], input_features[2].shape[3], para_k, para_b)
            aug_dec_feature_128 = model_128(input_features[2], offset)
        # 128+128->256 hybrid_feature_128
        hybrid_feature_128 = torch.cat([aug_dec_feature_128, input_features[2]], 1)
        # 384+256->640 x
        x += [hybrid_feature_128]
        x = torch.cat(x, 1)
        x = self.convs[("attentionConv", 2, 0)](x)
        # 640->512->256 x
        x = self.convs[("upconv", 2, 1)](x)
        x = self.convs[("upconv", 2, 2)](x)
        # 256->256  aug_enc_feature_256_2
        if flag:
            offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_256_2 = model_256(x, offset)
        else:
            offset = depth_to_scale(depth_4, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_256_2 = model_256(x, offset)
        # 256->128 x
        x = self.convs[("upconv", 2, 3)](x)
        # 256+128->384 x
        x = torch.cat([x, aug_enc_feature_256_2], 1)
        x = self.convs[("attentionConv", 2, 1)](x)
        # 384->256->128 x
        x = self.convs[("upconv", 2, 4)](x)
        x = self.convs[("upconv", 2, 5)](x)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](x))
        _, depth_3 = disp_to_depth(self.outputs[("disp", 3)], 0.1, 100)

        """
           第二次编码和解码特征融合增强
        """
        # 128->128 aug_enc_feature_128_1
        if flag:
            offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_128_1 = model_128(x, offset)
        else:
            offset = depth_to_scale(depth_3, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_128_1 = model_128(x, offset)
        # 128->64 x
        x = self.convs[("upconv", 3, 0)](x)
        # 128+64->192 x
        x = torch.cat([x, aug_enc_feature_128_1], 1)
        x = [upsample(x)]
        # 64->64 aug_dec_feature_64_0
        if flag:
            offset = depth_to_scale(depth_map, input_features[1].shape[2], input_features[1].shape[3], para_k, para_b)
            aug_dec_feature_64_0 = model_64(input_features[1], offset)
        else:
            offset = depth_to_scale(depth_3, input_features[1].shape[2], input_features[1].shape[3], para_k, para_b)
            aug_dec_feature_64_0 = model_64(input_features[1], offset)
        # 64+64->128 hybrid_feature_64_0
        hybrid_feature_64_0 = torch.cat([aug_dec_feature_64_0, input_features[1]], 1)
        # 128+192->320 x
        x += [hybrid_feature_64_0]
        x = torch.cat(x, 1)
        x = self.convs[("attentionConv", 3, 0)](x)
        # 320->256->128 x
        x = self.convs[("upconv", 3, 1)](x)
        x = self.convs[("upconv", 3, 2)](x)

        # 128->128 aug_enc_feature_128_2
        if flag:
            offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_128_2 = model_128(x, offset)
        else:
            offset = depth_to_scale(depth_3, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_128_2 = model_128(x, offset)
        # 128->64 x
        x = self.convs[("upconv", 3, 3)](x)
        # 128+64->192 x
        x = torch.cat([x, aug_enc_feature_128_2], 1)
        x = self.convs[("attentionConv", 3, 1)](x)
        # 192->128->64 x
        x = self.convs[("upconv", 3, 4)](x)
        x = self.convs[("upconv", 3, 5)](x)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](x))
        _, depth_2 = disp_to_depth(self.outputs[("disp", 2)], 0.1, 100)

        """
            第三次编码和解码特征融合增强
        """
        # 64->64 aug_enc_feature_64_1
        if flag:
            offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_64_1 = model_64(x, offset)
        else:
            offset = depth_to_scale(depth_2, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_64_1 = model_64(x, offset)
        # 64->32 x
        x = self.convs[("upconv", 4, 0)](x)
        # 64+32->96 x
        x = torch.cat([x, aug_enc_feature_64_1], 1)
        x = [upsample(x)]
        # 64->64 aug_dec_feature_64_1
        if flag:
            offset = depth_to_scale(depth_map, input_features[0].shape[2], input_features[0].shape[3], para_k, para_b)
            aug_dec_feature_64_1 = model_64(input_features[0], offset)
        else:
            offset = depth_to_scale(depth_2, input_features[0].shape[2], input_features[0].shape[3], para_k, para_b)
            aug_dec_feature_64_1 = model_64(input_features[0], offset)
        # 64+64->128 hybrid_feature_64_1
        hybrid_feature_64_1 = torch.cat([aug_dec_feature_64_1, input_features[0]], 1)
        # 128+96->224 x
        x += [hybrid_feature_64_1]
        x = torch.cat(x, 1)
        x = self.convs[("attentionConv", 4, 0)](x)
        # 224->128->64 x
        x = self.convs[("upconv", 4, 1)](x)
        x = self.convs[("upconv", 4, 2)](x)

        # 64->64 aug_enc_feature_64_2
        if flag:
            offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_64_2 = model_64(x, offset)
        else:
            offset = depth_to_scale(depth_2, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_64_2 = model_64(x, offset)
        # 64->32 x
        x = self.convs[("upconv", 4, 3)](x)
        # 64+32->96 x
        x = torch.cat([x, aug_enc_feature_64_2], 1)
        x = self.convs[("attentionConv", 4, 1)](x)
        # 96->64->32 x
        x = self.convs[("upconv", 4, 4)](x)
        x = self.convs[("upconv", 4, 5)](x)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](x))
        _, depth_1 = disp_to_depth(self.outputs[("disp", 1)], 0.1, 100)

        """
            第四次编码和解码特征融合增强
        """
        # 32->32 aug_enc_feature_32
        if flag:
            offset = depth_to_scale(depth_map, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_32 = model_32(x, offset)
        else:
            offset = depth_to_scale(depth_1, x.shape[2], x.shape[3], para_k, para_b)
            aug_enc_feature_32 = model_32(x, offset)
        # 32->16 x
        x = self.convs[("upconv", 5, 0)](x)
        # 32+16->48 x
        x = torch.cat([x, aug_enc_feature_32], 1)
        x = self.convs[("attentionConv", 5)](x)
        x = upsample(x)
        # 48->32->16 x
        x = self.convs[("upconv", 5, 1)](x)
        x = self.convs[("upconv", 5, 2)](x)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](x))
        return self.outputs
