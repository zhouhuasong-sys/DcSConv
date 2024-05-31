# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
import math


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_scale(depth, height, width, para_k):
    # 将深度图的尺寸采样到输入特征尺寸
    depth = torch.nn.functional.interpolate(depth, (height, width), mode="bilinear", align_corners=False)
    # 以卷积3*3和depth_mean为基准，先求得深度的相对比例，再通过sigmoid转换成scale
    factor = para_k / depth
    scale = factor * 3
    # 将scale[b,1,h,w]转换成offset[b,18,h,w]
    # 计算九个坐标的偏移量，假定中心点不变
    zeros = torch.zeros_like(scale)
    p_increment = scale.reshape(1, -1) / 2 - 1.5
    n_increment = p_increment * -1
    p_increment = p_increment.reshape(-1, 1, height, width)
    n_increment = n_increment.reshape(-1, 1, height, width)

    offset = torch.cat([n_increment, n_increment,
                        n_increment, zeros,
                        n_increment, p_increment,
                        zeros, n_increment,
                        zeros, zeros,
                        zeros, p_increment,
                        p_increment, n_increment,
                        p_increment, zeros,
                        p_increment, p_increment], dim=1)

    return offset, scale


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class Depth_Scaler(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_Scaler, self).__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.ELU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True)
        )

    def forward(self, raw_feature, depth, depth_mean):
        # 将深度图的尺寸采样到输入特征尺寸
        depth = torch.nn.functional.interpolate(depth, (raw_feature.shape[2], raw_feature.shape[3]),
                                                mode="bilinear", align_corners=False)
        # 将深度图转换为尺寸图
        factor = depth_mean / depth
        scale = 3 * factor

        # 计算三条分支的特征
        branch1 = self.branch1(raw_feature)
        branch2 = self.branch2(raw_feature)
        branch3 = self.branch3(raw_feature)

        # 计算三条分支的权重，距离越小，权重越高 [b, 1, h, w]
        dist_1 = torch.abs(scale - 1)
        dist_2 = torch.abs(scale - 3)
        dist_3 = torch.abs(scale - 5)
        weight_1 = math.e ** (-dist_1 ** 2 / 2 / 10 ** 2)
        weight_2 = math.e ** (-dist_2 ** 2 / 2 / 10 ** 2)
        weight_3 = math.e ** (-dist_3 ** 2 / 2 / 10 ** 2)

        # 归一化权重参数
        weight = torch.cat((weight_1, weight_2, weight_3), dim=1)
        prob = F.softmax(weight, dim=1)
        weight_nor_1 = prob[:, :1, :, :]
        weight_nor_2 = prob[:, 1:2, :, :]
        weight_nor_3 = prob[:, 2:, :, :]

        # 将对应分支的特征和权重图进行点乘
        feature_1 = torch.mul(branch1, weight_nor_1)
        feature_2 = torch.mul(branch2, weight_nor_2)
        feature_3 = torch.mul(branch3, weight_nor_3)

        # 相加融合得到增强特征
        feature = torch.add(feature_1, feature_2)
        feature = torch.add(feature, feature_3)

        return feature


class DepModule(nn.Module):
    def __init__(self, conv_3_1_in, conv_3_1_out, conv_3_2_in, conv_3_2_out, conv_1_in, conv_1_out, aug_enc):
        super(DepModule, self).__init__()
        self.conv_1 = ConvBlock(conv_3_1_in, conv_3_1_out)
        self.conv_2 = ConvBlock(conv_3_2_in, conv_3_2_out)
        self.conv_3 = ConvBlock_1(conv_1_in, conv_1_out)

        self.aug_1 = Depth_Scaler(conv_3_1_in, conv_3_1_in // 2)
        self.aug_2 = Depth_Scaler(aug_enc, aug_enc // 2)
        self.aug_3 = Depth_Scaler(conv_3_2_in, conv_3_2_in // 2)

    def forward(self, input_feature, encoder_feature, depth_map, para_k):
        # 第一次特征增强 256 -> 128
        aug_feature_fir = self.aug_1(input_feature, depth_map, para_k)
        # 正常卷积处理  256 -> 128
        input_feature = self.conv_1(input_feature)
        # 两个特征通道拼接上采样 128 + 128
        input_feature = torch.cat([input_feature, aug_feature_fir], 1)
        input_feature = [upsample(input_feature)]
        # 第二次特征增强 128 -> 64
        aug_enc_feature = self.aug_2(encoder_feature, depth_map, para_k)
        # 特征拼接  128+64+256 -> 448
        aug_hybrid_feature = torch.cat([aug_enc_feature, encoder_feature], 1)
        input_feature += [aug_hybrid_feature]
        input_feature = torch.cat(input_feature, 1)
        # 第三次特征增强 448 -> 224
        aug_feature_sec = self.aug_3(input_feature, depth_map, para_k)
        # 正常卷积处理  448 -> 224
        input_feature = self.conv_2(input_feature)
        # 两个特征通道拼接 224 + 224
        input_feature = torch.cat([input_feature, aug_feature_sec], 1)
        # 第三次通道注意力融合
        # 1*1卷积降低特征通道数 448 -> 128
        feature = self.conv_3(input_feature)
        return feature


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channel, channel // 16, kernel_size=1, stride=1, padding=0),
                                nn.ReLU(),
                                nn.Conv2d(channel // 16, channel, kernel_size=1, stride=1, padding=0))
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
                                   nn.Sigmoid())

    def forward(self, conv_feature, aug_feature, scale_map):
        features = torch.cat([conv_feature, aug_feature], dim=1)
        avg_out = self.fc(self.avg_pool(features))
        max_out = self.fc(self.max_pool(features))
        attention_1 = self.sigmoid(avg_out + max_out)

        features_c = features * attention_1

        avg_out = torch.mean(features_c, dim=1, keepdim=True)
        max_out, _ = torch.max(features_c, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        attention_2 = self.conv1(x)
        out = features_c * attention_2
        return out


class SAF2(nn.Module):
    def __init__(self, channel):
        super(SAF2, self).__init__()
        self.global_path = nn.Sequential(nn.Conv2d(channel + 1, channel, kernel_size=1, stride=1, padding=0),
                                         nn.ReLU(True),
                                         nn.AdaptiveAvgPool2d(1),
                                         nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                         nn.ReLU(True),
                                         nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid())
        self.conv1 = nn.Sequential(nn.Conv2d(channel + 1, channel, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
                                   nn.Sigmoid())

        self.conv3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(True))

    def forward(self, conv_feature, aug_feature, scale_map):
        features = torch.cat([conv_feature, aug_feature], dim=1)
        scale_map = (scale_map - 3) / 3

        x = torch.cat([features, scale_map], dim=1)
        attention_1 = self.global_path(x)

        y = features * attention_1.expand_as(features)
        y = torch.cat([y, scale_map], dim=1)
        y = self.conv1(y)
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out, _ = torch.max(y, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        attention_2 = self.conv2(y)

        aug_feature = aug_feature * attention_2
        mid = torch.cat([conv_feature, aug_feature], dim=1)
        out = features + mid * attention_1.expand_as(mid)
        return self.conv3(out)


class DepModule_4(nn.Module):
    def __init__(self, conv_3_1_in, conv_3_1_out, conv_3_2_in, conv_3_2_out, conv_1_in, conv_1_out, aug_enc):
        super(DepModule_4, self).__init__()
        self.conv_1 = ConvBlock(conv_3_1_in, conv_3_1_out)
        self.conv_2 = ConvBlock(conv_3_2_in, conv_3_2_out)
        self.conv_3 = ConvBlock_1(conv_1_in, conv_1_out)

        self.aug_1 = ops.DeformConv2d(conv_3_1_in, conv_3_1_in // 2, kernel_size=3, padding=1, stride=1)
        self.aug_2 = ops.DeformConv2d(aug_enc, aug_enc // 2, kernel_size=3, padding=1, stride=1)
        self.aug_3 = ops.DeformConv2d(conv_3_2_in, conv_3_2_in // 2, kernel_size=3, padding=1, stride=1)

        self.attention_1 = SAF2(conv_3_1_out * 2)
        self.attention_2 = SAF2(aug_enc + aug_enc // 2)
        self.attention_3 = SAF2(conv_3_2_out * 2)

    def forward(self, input_feature, offset_small, encoder_feature, offset_large, scale_small, scale_large):
        # 第一次特征增强 256 -> 128
        aug_feature_fir = self.aug_1(input_feature, offset_small)
        # 正常卷积处理  256 -> 128
        input_feature = self.conv_1(input_feature)
        # 两个特征通道拼接上采样 128 + 128 -> 256
        # 第一次通道注意力融合
        input_feature = self.attention_1(input_feature, aug_feature_fir, scale_small)
        input_feature = [upsample(input_feature)]

        # 第二次特征增强 128 -> 64
        aug_enc_feature = self.aug_2(encoder_feature, offset_large)
        # 特征拼接  128+64+256 -> 448
        # 第二次通道注意力融合
        aug_hybrid_feature = self.attention_2(encoder_feature, aug_enc_feature, scale_large)

        input_feature += [aug_hybrid_feature]
        input_feature = torch.cat(input_feature, 1)
        # 第三次特征增强 448 -> 224
        aug_feature_sec = self.aug_3(input_feature, offset_large)
        # 正常卷积处理  448 -> 224
        input_feature = self.conv_2(input_feature)
        # 两个特征通道拼接 224 + 224
        # 第三次通道注意力融合
        input_feature = self.attention_3(input_feature, aug_feature_sec, scale_large)
        # 1*1卷积降低特征通道数 448 -> 128
        feature = self.conv_3(input_feature)
        return feature


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock_1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
