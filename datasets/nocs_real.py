#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import random
import hydra
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
import matplotlib.pyplot as plt
from os import makedirs, remove
from os.path import exists, join
import glob
import os.path
import json
import pickle
import h5py
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
from multiprocessing import Manager

import scipy.io as sio
from scipy.spatial.transform import Rotation as sciR

import __init__
from global_info import global_info
from common.d3_utils import align_rotation, rotate_about_axis, transform_pcloud
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np, label_relative_rotation_np

def bp():
    import pdb;pdb.set_trace()

infos           = global_info()
project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories
second_path     = infos.second_path

def get_index(src_length, tgt_length):
    idx = np.arange(0, src_length)
    if src_length < tgt_length:
        idx = np.pad(idx, (0, tgt_length - src_length), 'wrap')
    idx = np.random.permutation(idx)[:tgt_length]
    return idx

class NOCSDatasetReal(data.Dataset):
    def __init__(self, cfg, split='train', root=None):
        if root is None:
            self.root = cfg.DATASET.data_path
        else:
            self.root  = root
        self.cfg = cfg
        self.task = cfg.task
        if cfg.DATASET.real:
            if split == 'train':
                split = f'real_{split}'
            elif split == 'val':
                split = 'real_test'
        self.split = split
        self.npoints       = cfg.out_points
        self.out_points    = cfg.out_points # for reconstruction
        if self.cfg.use_fps_points:
            self.out_points    = 4 * cfg.out_points
            self.npoints       = 4 * cfg.out_points

        self.anchors = L.get_anchors()

        self.category = cfg.category
        dir_point = join(self.root, split, str(self.category))
        self.datapath = sorted(glob.glob(f'{dir_point}/*/*/*/*.npz'))

        category_id  = categories[self.category]
        if 'train' in split:
            split_folder = 'train'
        else:
            split_folder = 'val'
        self.instance_path = join(self.root, 'model_pts')
        self.object_path = join(second_path, f'data/nocs/obj_models/{split_folder}', category_id)
        self.object_path_external = join(group_path, f'external/ShapeNetCore.v2/{category_id}/')
        instances_used = [f for f in os.listdir(dir_point) if os.path.isdir(join(dir_point, f))]
        instances = [f for f in os.listdir(self.object_path_external) if os.path.isdir(join(self.object_path_external, f))]
        print('--checking ', self.object_path, f' with {len(instances_used)} instances')
        print('--checking ', self.object_path_external, f' with {len(instances)} instances')
        self.instance_points = {}
        self.instances = instances
        for instance in instances_used:
            model_path = join(self.object_path_external, instance, 'models', 'surface_points.pkl')
            if os.path.exists(model_path):
                with open(model_path, "rb") as obj_f:
                    self.instance_points[instance] = pickle.load(obj_f)
            else:
                self.instance_points[instance] = np.load(join(self.instance_path, f'{instance}.npy'))

        # create NOCS dict
        manager = Manager()
        self.nocs_dict = manager.dict()
        self.cls_dict = manager.dict()
        self.cloud_dict = manager.dict()
        self.g_dict    = manager.dict()
        self.r_dict    = manager.dict()

        if self.cfg.eval or self.split != 'train':
            np.random.seed(0)

        self.scale_dict = {'bottle': 0.5, 'bowl': 0.25, 'camera': 0.27,
                           'can': 0.2, 'laptop': 0.5, 'mug': 0.21}
        self.scale_factor = self.scale_dict[cfg.category.split('_')[0]]

    def get_complete_cloud(self, instance):
        if instance in self.instance_points:
            pts = self.instance_points[instance]
        else:
            pts = self.instance_points[self.instances[0]]
        idx = get_index(len(pts), self.out_points)
        return pts[idx]

    def get_sample_partial(self, idx, verbose=False):
        fn = self.datapath[idx]
        if verbose:
            print(fn)
        category_name = fn.split('.')[-2].split('/')[-5]
        instance_name = fn.split('.')[-2].split('/')[-4] + '_' + fn.split('.')[-2].split('/')[-3] + '_' + fn.split('.')[-2].split('/')[-1]
        instance_id   = fn.split('.')[-2].split('/')[-4]

        model_points  = self.get_complete_cloud(instance_id)
        if self.category == 'laptop':
            model_points = model_points  + 0.5
        else:
            boundary_pts = [np.min(model_points, axis=0), np.max(model_points, axis=0)]
            center_pt = (boundary_pts[0] + boundary_pts[1])/2
            length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
            model_points = (model_points - center_pt.reshape(1, 3))/length_bb
            calib_mat = rotate_about_axis(90 / 180 * np.pi, axis='y')
            model_points = model_points @ calib_mat  + 0.5

        data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
        p_arr = data_dict['points']
        pose = data_dict['pose']
        r, t, s = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
        if self.cfg.normalize_scale:
            scale_normalize = s
        else:
            scale_normalize = self.scale_factor
        n_arr = np.matmul(p_arr - t, r) / s + 0.5
        if verbose:
            print(f'we have {p_arr.shape[0]} pts')
        full_points = np.concatenate([p_arr, n_arr], axis=1)
        full_points = np.random.permutation(full_points)

        idx = get_index(len(full_points), self.npoints)
        pos = torch.from_numpy(full_points[idx, :3].astype(np.float32))
        nocs_gt = torch.from_numpy(full_points[idx, 3:6].astype(np.float32))

        T = torch.from_numpy(t.astype(np.float32))
        S = torch.from_numpy(np.array([s]).astype(np.float32))
        _, R_label, R0 = rotation_distance_np(r, self.anchors)
        R_gt = torch.from_numpy(r.astype(np.float32)) # predict r

        if self.cfg.pre_compute_delta:
            xyz = nocs_gt - 0.5
        else:
            xyz = pos/scale_normalize
            T   = T/scale_normalize
            model_points = (model_points - 0.5) * s / scale_normalize + 0.5 # a scaled version
            nocs_gt = (nocs_gt - 0.5) * s / scale_normalize + 0.5
            # xyz = sR * P + T
            # xyz/scale = R * (s*P/scale) + T/scale

        return {'xyz': xyz,
                'points': torch.from_numpy(model_points.astype(np.float32)),  # replace nocs_gt as new target
                'full': torch.from_numpy(model_points.astype(np.float32)),    # complete point cloud
                'label': torch.from_numpy(np.array([1]).astype(np.float32)),
                'R_gt': R_gt,
                'R_label': R_label,
                'R': R0,
                'T': T,
                'S': S,
                'fn': fn,
                'id': instance_name,
                'idx': idx,
                'class': category_name
               }

    def __getitem__(self, idx, verbose=False):
        sample = self.get_sample_partial(idx, verbose=verbose)
        return sample

    def __len__(self):
        return len(self.datapath)

def check_data(data_dict):
    print(f'path: {data_dict["fn"]}, class: {data_dict["class"]}, instance: {data_dict["id"]}')
    cloud, canon_cloud, full = data_dict['xyz'], data_dict['points'], data_dict['full']
    S, R, T = data_dict['S'].numpy(), data_dict['R_gt'].numpy(), data_dict['T'].numpy()
    posed_canon_cloud = np.dot(canon_cloud - 0.5, R.T) + T
    posed_full_cloud  = np.dot(full - 0.5, R.T) + T
    num_plots = 3
    plt.figure(figsize=(6 * num_plots, 6))

    def plot(ax, pt_list, title):
        all_pts = np.concatenate(pt_list, axis=0)
        pmin, pmax = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2
        for pts in pt_list:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=3**2)
        ax.set_xlim3d([center[0] - lim, center[0] + lim])
        ax.set_ylim3d([center[1] - lim, center[1] + lim])
        ax.set_zlim3d([center[2] - lim, center[2] + lim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)

    for i, (name, pt_list) in enumerate(zip(
            ['partial', 'canon_partial_and_complete', 'posed_canon_partial_and_complete'],
            [[cloud], [canon_cloud, full], [posed_canon_cloud, posed_full_cloud]])):
        ax = plt.subplot(1, num_plots, i + 1, projection='3d')
        plot(ax, pt_list, name)

    plt.show()

@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False) #

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30) #
    cfg.item     ='nocs_real'
    cfg.name_dset='nocs_real'
    cfg.log_dir  = infos.second_path + cfg.log_dir
    # dset = NOCSDatasetReal(cfg=cfg, split='train')
    dset = NOCSDatasetReal(cfg=cfg, split='train')
    #
    for i in range(len(dset)):  #
        dp = dset.__getitem__(i, verbose=True)
        print(f'--checking {i}th data')
        check_data(dp)

if __name__ == '__main__':
    main()
    # python nocs_synthetic.py category='5' datasets=nocs_synthetic
