from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import math
from layers import *
import statistics

class Depth_scaler(nn.Module):
    def __init__(self, channels):
        super(Depth_scaler, self).__init__()

        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    #    [batch, channel, height, width]
    def forward(self, raw_feature, depth_map, device):
        # 给定三个阈值 constant

        first_threshold = 0.3
        second_threshold = 0.7
        third_threshold = 10

        """
        # 计算三个阈值 batch
        pre_depth_numpy = depth_map.cpu().detach().numpy()
        depth_map = depth_map.to(device)
        first_threshold = np.percentile(pre_depth_numpy, 25)
        second_threshold = np.percentile(pre_depth_numpy, 50)
        third_threshold = np.percentile(pre_depth_numpy, 75)
        """
        # 计算四条分支的特征
        branch1 = self.branch1(raw_feature)
        branch2 = self.branch2(raw_feature)
        branch3 = self.branch3(raw_feature)
        branch4 = self.branch4(raw_feature)

        # 将深度图的尺寸采样到输入特征尺寸
        depth_map = torch.nn.functional.interpolate(depth_map, (raw_feature.shape[2], raw_feature.shape[3]),
                                                    mode="bilinear", align_corners=False)

        # 计算每路分支的掩膜mask
        depth_residual_1 = depth_map - first_threshold
        depth_residual_2 = depth_map - second_threshold
        depth_residual_3 = depth_map - third_threshold

        depth_mask_1 = torch.where(depth_residual_1 < 0, 1, 0)
        depth_mask_2 = torch.where(torch.mul(depth_residual_1,depth_residual_2) <= 0, 1, 0)
        depth_mask_3 = torch.where(torch.mul(depth_residual_2, depth_residual_3) <= 0, 1, 0)
        depth_mask_4 = torch.where(depth_residual_3 > 0, 1, 0)

        # 将对应分支的特征和掩膜图进行点乘
        feature_1 = torch.mul(branch1, depth_mask_1)
        feature_2 = torch.mul(branch2, depth_mask_2)
        feature_3 = torch.mul(branch3, depth_mask_3)
        feature_4 = torch.mul(branch4, depth_mask_4)

        """
        # 计算每路分支的权重图
        first_mean = first_threshold - 0.1
        second_mean = second_threshold - first_threshold
        third_mean = third_threshold - second_threshold
        fourth_mean = 100 - third_threshold

        first_weight = 100 - abs(depth_map - first_mean)
        second_weight = 100 - abs(depth_map - second_mean)
        third_weight = 100 - abs(depth_map - third_mean)
        fourth_weight = 100 - abs(depth_map - fourth_mean)

        weight_sum = first_weight + second_weight + third_weight + fourth_weight
        first_weight = first_weight / weight_sum
        second_weight = second_weight / weight_sum
        third_weight = third_weight / weight_sum
        fourth_weight = fourth_weight / weight_sum
        
        # 将对应分支的特征和权重图进行点乘
        feature_1 = torch.mul(branch1, first_weight)
        feature_2 = torch.mul(branch2, second_weight)
        feature_3 = torch.mul(branch3, third_weight)
        feature_4 = torch.mul(branch4, fourth_weight)
        """
        # 相加融合得到增强特征
        feature = torch.add(feature_1, feature_2)
        feature = torch.add(feature, feature_3)
        feature = torch.add(feature, feature_4)

        return feature
