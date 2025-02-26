# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from np_transformation import SEs2kittiformat, SEs2ses

import sys
sys.path.append('../../BotVIO')
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from util.utils import readlines, relative2absolute
from options import BotVIOOptions
from datasets import KITTIOdomDataset
import networks
from tqdm import tqdm
import time

from thop import clever_format
from thop import profile

def profile_pose(encoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)

    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")

    return flops_e, params_e

def profile_posed(decoder, v):
    flops_d, params_d = profile(decoder, inputs=(v[0,:].unsqueeze(0),), verbose=False)
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")
    return flops_d, params_d
    
def profile_imu(encoder, x):
    flops_e, params_e = profile(encoder, inputs=(x[0, :, :].unsqueeze(0), ), verbose=False)
    flops_ie, params_ie = clever_format([flops_e, params_e], "%.3f")
    return flops_ie, params_ie
    
def profile_fusion(encoder, x, y):
    flops_e, params_e = profile(encoder, inputs=(x[0,:].unsqueeze(0), y[0,:].unsqueeze(0),), verbose=False)
    flops_ie, params_ie = clever_format([flops_e, params_e], "%.3f")
    return flops_ie, params_ie


def evaluate(opt):
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    for sequence_id in range(9, 11):
        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "../splits", "odom",
                         "test_files_{:02d}.txt".format(sequence_id)))

        dataset = KITTIOdomDataset(opt.eval_data_path, filenames, opt.height, opt.width,
                                   [0, 1], 8, is_train=False, img_ext='.jpg')
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        pose_encoder = networks.PoseEncoder()
        pose_decoder = networks.Pose_CNN(opt)
        imu_encoder = networks.InertialEncoder()
        fusion = networks.Trans_Fusion(dim = opt.v_f_len)

        pose_encoder.cuda().half()
        pose_encoder.eval()
        pose_decoder.cuda().half()
        pose_decoder.eval()
        
        imu_encoder.cuda().half()
        imu_encoder.eval()
        fusion.cuda()
        fusion.eval()

        pred_local_mat = []

        opt.frame_ids = [0, 1]  # pose network only takes two frames as input
        t_mean = 0
        t_toal = 0
        len_s = len(dataloader)
        bar = tqdm(range(len(dataloader)))
        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    if isinstance(ipt, list):
                        continue
                    inputs[key] = ipt.cuda().half()

                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)
                imus = inputs["imu"]
                t1 = time.time()
                features = pose_encoder(all_color_aug)
                imu = imu_encoder(imus)
                fused = fusion(features.float(), imu.float())
                axisangle, translation = pose_decoder(fused.half())
                
                t2 = time.time()
                t_toal += (t2-t1)
                print("running time of per frame ", (t2-t1)/12*1000)
                
                flops_e, params_e = profile_pose(pose_encoder, all_color_aug)
                print("visual_encoder",flops_e, params_e)
                flops_i, params_i = profile_imu(imu_encoder, imus)
                print("imu_encoder",flops_i, params_i)
                flops_f, params_f = profile_fusion(fusion, features, imu)
                print("fusion",flops_f, params_f)
                flops_d, params_d = profile_posed(pose_decoder, fused.half())
                print("pose_decoder", flops_d, params_d)

                pred_local_mat.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                bar.update()
        t_mean = t_toal/len_s/opt.batch_size*1000

        print("mean running time", t_mean)

if __name__ == "__main__":
    options = BotVIOOptions()
    evaluate(options.parse())
