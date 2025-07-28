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


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    for sequence_id in range(9, 11):
        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "../splits", "odom",
                         "test_files_{:02d}.txt".format(sequence_id)))

        dataset = KITTIOdomDataset(opt.eval_data_path, filenames, opt.height, opt.width,
                                   [0, 1], 4, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, 2, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        pose_encoder_path = os.path.join(opt.load_weights_folder, "visual_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose_decoder.pth")
        imu_encoder_path = os.path.join(opt.load_weights_folder, "imu_encoder.pth")
        fusion_path = os.path.join(opt.load_weights_folder, "fusion.pth")
        
        pose_encoder = networks.PoseEncoder()
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.Pose_CNN(opt)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        imu_encoder = networks.InertialEncoder()
        imu_encoder.load_state_dict(torch.load(imu_encoder_path))
        
        fusion = networks.Trans_Fusion(dim=opt.v_f_len)
        fusion.load_state_dict(torch.load(fusion_path))

        pose_encoder.cuda().half()
        pose_encoder.eval()
        pose_decoder.cuda().half()
        pose_decoder.eval()
        fusion.cuda().half()
        fusion.eval()
        imu_encoder.cuda().half()
        imu_encoder.eval()

        pred_local_mat = []

        print("sequence_id", sequence_id)
        print("-> Computing pose predictions")

        opt.frame_ids = [0, 1]  # pose network only takes two frames as input

        bar = tqdm(range(len(dataloader)))
        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    if isinstance(ipt, list):
                        continue
                    inputs[key] = ipt.cuda().half()

                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)
                imus = inputs["imu"]
                
                visual = pose_encoder(all_color_aug)
                imu = imu_encoder(imus)
                fused = fusion(visual, imu)
                axisangle, translation = pose_decoder(fused)

                pred_local_mat.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                bar.update()

        pred_local_mat = np.concatenate(pred_local_mat)
        pred_global_mat = relative2absolute(pred_local_mat)

        gt_poses_path = os.path.join(opt.eval_data_path, "poses", "{:02d}.txt".format(sequence_id))
        gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
        gt_global_poses = np.concatenate(
            (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
        gt_global_poses[:, 3, 3] = 1
        gt_xyzs = gt_global_poses[:, :3, 3]

        gt_local_poses = []
        for i in range(1, len(gt_global_poses)):
            gt_local_poses.append(
                np.linalg.inv(gt_global_poses[i]) @ gt_global_poses[i-1])

        ates = []
        # num_frames = gt_xyzs.shape[0]
        num_frames = gt_xyzs.shape[1]#原代码似乎有问题～
        track_length = 5
        for i in range(0, num_frames - 1):
            local_xyzs = np.array(dump_xyz(pred_local_mat[i:i + track_length - 1]))
            gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

            ates.append(compute_ate(gt_local_xyzs, local_xyzs))

        print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

        pred_global_poses = SEs2kittiformat(pred_global_mat)

        save_path = os.path.join(opt.load_weights_folder, f"odom_{sequence_id:02}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(opt.load_weights_folder, f"odom_{sequence_id:02}/poses.npy")
        np.save(save_path, pred_local_mat)
        print("-> Predictions saved to", save_path)

        save_path = f"./results/{sequence_id:02}.txt"
        np.savetxt(save_path, pred_global_poses)
        print("-> Pred Global Poses saved to", save_path)

if __name__ == "__main__":
    options = BotVIOOptions()
    evaluate(options.parse())
