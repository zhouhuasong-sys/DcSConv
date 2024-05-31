# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import readlines
from evaluate_depth import STEREO_SCALE_FACTOR

import warnings
from kitti_utils import generate_depth_map

from torch.utils.data import DataLoader
from options import MonodepthOptions
import datasets

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
file_dir2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path',
                        type=str,
                        help='path to a test image or folder of images',
                        default="/data1/raw_data/data")
    parser.add_argument('--pre_model_path',
                        type=str,
                        help='name of a pre-training model to use',
                        default="/data/users/huasong/monodepth2-z/models/mono_640x192")
    parser.add_argument('--model_path',
                        type=str,
                        help='name of a model to use',
                        default="/data1/raw_data/mono_offset/mdp/models/weights_2")
    parser.add_argument('--ext',
                        type=str,
                        help='image extension to search for in folder',
                        default="png")

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    # 加载monodepth2的预训练模型
    mono_encoder_path = os.path.join(args.pre_model_path, "encoder.pth")
    mono_depth_decoder_path = os.path.join(args.pre_model_path, "depth.pth")

    mono_encoder = networks.ResnetEncoder(18, False)
    mono_loaded_dict_enc = torch.load(mono_encoder_path)
    feed_height = mono_loaded_dict_enc['height']
    feed_width = mono_loaded_dict_enc['width']
    mono_filtered_dict_enc = {k: v for k, v in mono_loaded_dict_enc.items() if k in mono_encoder.state_dict()}
    mono_encoder.load_state_dict(mono_filtered_dict_enc)
    mono_encoder.cuda()
    mono_encoder.eval()

    mono_depth_decoder = networks.DepthDecoder(num_ch_enc=mono_encoder.num_ch_enc, scales=range(4))
    mono_loaded_dict = torch.load(mono_depth_decoder_path)
    mono_depth_decoder.load_state_dict(mono_loaded_dict)
    mono_depth_decoder.cuda()
    mono_depth_decoder.eval()

    # 加载网络训练的模型
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    decoder_path = os.path.join(args.model_path, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.Deformable_DepthDecoder()

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = os.path.join(args.image_path, "result")
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    # 获取真值图
    gt_depths = []
    lines = []
    line_0 = "2011_09_26/2011_09_26_drive_0002_sync 0000000057 l"
    line_1 = "2011_09_26/2011_09_26_drive_0002_sync 0000000054 l"
    line_2 = "2011_09_26/2011_09_26_drive_0002_sync 0000000069 l"
    line_3 = "2011_09_26/2011_09_26_drive_0002_sync 0000000042 l"
    lines.append(line_0)
    lines.append(line_1)
    lines.append(line_2)
    lines.append(line_3)
    for line in lines:
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)
        calib_dir = os.path.join(os.path.join(file_dir2, "/data/Data/raw_data"), folder.split("/")[0])
        velo_filename = os.path.join(os.path.join(file_dir2, "/data/Data/raw_data"), folder,
                                     "velodyne_points/data", "{:010d}.bin".format(frame_id))
        gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        gt_depths.append(gt_depth)

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.cuda()

            pre_depth_outputs = mono_depth_decoder(mono_encoder(input_image))
            pre_disp = pre_depth_outputs[("disp", 0)]
            _, pre_depth = disp_to_depth(pre_disp, 0.1, 100)

            features = encoder(input_image)
            outputs = depth_decoder(features, True, pre_depth, 0.6477)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving colormapped depth image
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            disp_resized_np = disp_resized_np - gt_depths[idx]
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
