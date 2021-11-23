#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import hydra
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf

import glob

import json
import h5py
import pickle
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
from multiprocessing import Manager

import scipy.io as sio
from scipy.spatial.transform import Rotation as sciR
import matplotlib.pyplot as plt
import os
from os import makedirs, remove
from os.path import exists, join
import os.path

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

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def get_index(src_length, tgt_length):
    idx = np.arange(0, src_length)
    if src_length < tgt_length:
        idx = np.pad(idx, (0, tgt_length - src_length), 'wrap')
    idx = np.random.permutation(idx)[:tgt_length]
    return idx

class NOCSDataset(data.Dataset):
    def __init__(self, cfg, split='train', root=None, npoint_shift=False, is_testing=False, rand_seed=999):
        if root is None:
            self.root = cfg.DATASET.data_path
        else:
            self.root  = root
        self.cfg   = cfg
        self.task  = cfg.task
        self.split = split
        # needed number
        self.npoints       = cfg.out_points
        self.out_points    = cfg.out_points # for reconstruction
        if self.cfg.use_fps_points:
            self.out_points    = 4 * cfg.out_points
            self.npoints       = 4 * cfg.out_points

        self.num_gen_samples=cfg.DATASET.num_gen_samples
        self.num_of_class  =cfg.DATASET.num_of_class
        self.radius = 0.1
        self.num_samples = 10 # default to be 10

        # augmentation
        self.augment   = cfg.augment

        if self.task == 'category_pose':
            self.is_debug     = cfg.is_debug

        self.is_testing   = is_testing
        shape_ids = {}

        self.anchors = L.get_anchors()

        assert(split == 'train' or split == 'val')
        self.category = cfg.category
        dir_point = join(self.root, split, str(self.category))
        self.datapath = sorted(glob.glob(f'{dir_point}/*/*/*/*.npz'))

        category_id  = categories[self.category]
        self.object_path = join(second_path, f'data/nocs/obj_models/{split}', category_id)
        self.object_path_external = join(group_path, f'external/ShapeNetCore.v2/{category_id}/')
        instances_used = [f for f in os.listdir(self.object_path_external) if os.path.isdir(join(self.object_path, f))]
        instances = [f for f in os.listdir(self.object_path_external) if os.path.isdir(join(self.object_path_external, f))]
        print('--checking ', self.object_path, f' with {len(instances_used)} instances')
        print('--checking ', self.object_path_external, f' with {len(instances)} instances')
        self.instance_points = {}
        for instance in instances:
            model_path = join(self.object_path_external, instance, 'models', 'surface_points.pkl')
            with open(model_path, "rb") as obj_f:
                self.instance_points[instance] = pickle.load(obj_f)

        # create NOCS dict
        manager = Manager()
        self.nocs_dict = manager.dict()
        self.cls_dict = manager.dict()
        self.cloud_dict = manager.dict()
        self.g_dict    = manager.dict()
        self.r_dict    = manager.dict()
        self.scale_dict = {'bottle': 0.5, 'bowl': 0.25, 'camera': 0.27,
                           'can': 0.2, 'laptop': 0.5, 'mug': 0.21}
        self.scale_factor = self.scale_dict[cfg.category.split('_')[0]]

        # pre-fetch
        self.backup_cache = []
        for j in range(300):
            fn  = self.datapath[j]
            category_name = fn.split('.')[-2].split('/')[-5]
            instance_name = fn.split('.')[-2].split('/')[-4] + '_' + fn.split('.')[-2].split('/')[-3] + '_' + fn.split('.')[-2].split('/')[-1]
            data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
            labels    = data_dict['labels']
            p_arr     = data_dict['points'][labels]
            if p_arr.shape[0] > 256:
                self.backup_cache.append([category_name, instance_name, data_dict, j])
        print('backup has ', len(self.backup_cache))

    def get_complete_cloud(self, instance):
        pts = self.instance_points[instance]
        idx = get_index(len(pts), self.out_points)
        return pts[idx]

    def get_sample_partial(self, idx, verbose=False):
        fn  = self.datapath[idx]
        category_name = fn.split('.')[-2].split('/')[-5]
        instance_name = fn.split('.')[-2].split('/')[-4] + '_' + fn.split('.')[-2].split('/')[-3] + '_' + fn.split('.')[-2].split('/')[-1]
        instance_id   = fn.split('.')[-2].split('/')[-4]
        model_points  = self.get_complete_cloud(instance_id)

        data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
        labels    = data_dict['labels']
        p_arr     = data_dict['points'][labels]
        if p_arr.shape[0] < 100:
            category_name, instance_name, data_dict, idx = self.backup_cache[random.randint(0, len(self.backup_cache)-1)]
            labels = data_dict['labels']
            p_arr  = data_dict['points'][labels]
            model_points = self.get_complete_cloud(instance_name.split('_')[0])

        boundary_pts = [np.min(model_points, axis=0), np.max(model_points, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        model_points = (model_points - center_pt.reshape(1, 3))/length_bb
        calib_mat = rotate_about_axis(90 / 180 * np.pi, axis='y')
        model_points = model_points @ calib_mat  + 0.5

        # load full points
        rgb       = data_dict['rgb'][labels] / 255.0
        pose      = data_dict['pose']
        r, t, s   = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
        labels = labels[labels].astype(np.int).reshape(-1, 1) # only get true
        if s < 0.00001:
            s = 1
        n_arr     = np.matmul(p_arr - t, r) / s + 0.5 # scale
        center    = t.reshape(1, 3)
        bb_pts    = np.array([[0.5, 0.5, 0.5]])
        bb_pts    = s * np.matmul(bb_pts, r.T)  + t.reshape(1, 3) # we actually don't know the exact bb, sad
        center_offset = p_arr - center #
        bb_offset =  bb_pts - p_arr #
        up_axis   = np.matmul(np.array([[0.0, 1.0, 0.0]]), r.T)
        if verbose:
            print(f'we have {p_arr.shape[0]} pts')
        full_points = np.concatenate([p_arr, n_arr, rgb], axis=1)
        full_points = np.random.permutation(full_points)

        idx         = get_index(len(full_points), self.npoints)
        pos         = torch.from_numpy(full_points[idx, :3].astype(np.float32)).unsqueeze(0)
        nocs_gt     = torch.from_numpy(full_points[idx, 3:6].astype(np.float32))

        if self.cfg.MODEL.num_in_channels == 1:
            feat = torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
        elif self.cfg.use_rgb:
            feat = torch.from_numpy(full_points[idx, 6:9].astype(np.float32)).unsqueeze(0)
        else:
            feat = torch.from_numpy(full_points[idx, 6:9].astype(np.float32)).unsqueeze(0)

        T = torch.from_numpy(t.astype(np.float32))
        S = torch.from_numpy(np.array([s]).astype(np.float32))
        _, R_label, R0 = rotation_distance_np(r, self.anchors)
        R_gt = torch.from_numpy(r.astype(np.float32)) # predict r
        scale_normalize = self.scale_factor
        if self.cfg.pre_compute_delta:
            xyz = nocs_gt - 0.5
        else:
            xyz = pos[0]/scale_normalize
            T   = T/scale_normalize
            model_points = (model_points - 0.5) * s / scale_normalize + 0.5 # a scaled version
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

    def get_sample_full(self, idx, verbose=False):
        fn  = self.datapath[idx]
        category_name = fn.split('.')[0].split('/')[-5]
        instance_name = fn.split('.')[0].split('/')[-4] + '_' + fn.split('.')[0].split('/')[-3] + '_' + fn.split('.')[0].split('/')[-1]

        data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
        labels    = data_dict['labels']
        p_arr     = data_dict['points']
        if p_arr.shape[0] < self.npoints:
            category_name, instance_name, data_dict, idx = self.backup_cache[random.randint(0, len(self.backup_cache)-1)]
            labels = data_dict['labels']
            p_arr  = data_dict['points']
        ind_fore  = np.where(labels)[0]
        ind_back  = np.where(~labels)[0]

        rgb       = data_dict['rgb'] / 255.0
        pose      = data_dict['pose']
        r, t, s   = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
        labels    = labels.astype(np.int)
        if s < 0.00001:
            s = 1
        n_arr     = np.matmul(p_arr - t, r) / s + 0.5 # scale
        center    = t.reshape(1, 3)
        bb_pts    = np.array([[0.5, 0.5, 0.5]])
        bb_pts    = s * np.matmul(bb_pts, r.T)  + t.reshape(1, 3) # we actually don't know the exact bb, sad
        center_offset = p_arr - center #
        bb_offset =  bb_pts - p_arr #
        up_axis   = np.matmul(np.array([[0.0, 1.0, 0.0]]), r.T)

        # choose inds
        half_num    = int(self.npoints/2)
        if len(ind_back) < half_num:
            half_num = self.npoints - len(ind_back)
        fore_choose = np.random.permutation(ind_fore)[:half_num]
        another_half_num = self.npoints - len(fore_choose)
        back_choose = np.random.permutation(ind_back)[:another_half_num]
        all_inds    = np.concatenate([fore_choose, back_choose])
        pos         = torch.from_numpy(p_arr[all_inds].astype(np.float32)).unsqueeze(0)
        nocs_gt     = torch.from_numpy(n_arr[all_inds].astype(np.float32))
        labels      = torch.from_numpy(labels[all_inds].astype(np.float32)) # N

        if self.cfg.MODEL.num_in_channels == 1:
            feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
        else:
            feat      = torch.from_numpy(rgb[all_inds].astype(np.float32)).unsqueeze(0)

        T = torch.from_numpy(t.astype(np.float32))
        _, R_label, R0 = rotation_distance_np(r, self.anchors)
        R_gt = torch.from_numpy(r.astype(np.float32)) # predict r
        center = torch.from_numpy(np.array([[0.5, 0.5, 0.5]])) # 1, 3
        center_offset = pos[0].clone().detach() - T #

        return {'pc': (pos[0] - T),
                'label': torch.from_numpy(np.array([1])).long(),
                'R': R0,
                'id': idx,
                'R_gt' : R_gt,
                'R_label': torch.Tensor([R_label]).long(),
               }

    def __getitem__(self, idx, verbose=False):
        if self.cfg.use_background:
            sample = self.get_sample_full(idx, verbose=verbose)
        else:
            sample = self.get_sample_partial(idx, verbose=verbose)
        return sample

    def __len__(self):
        return len(self.datapath)

def check_data(data_dict):
    print(f'path: {data_dict["fn"]}, class: {data_dict["class"]}, instance: {data_dict["id"]}')
    cloud, canon_cloud, full = data_dict['xyz'], data_dict['points'], data_dict['full']
    S, R, T = data_dict['S'].numpy(), data_dict['R_gt'].numpy(), data_dict['T'].numpy()
    posed_canon_cloud = S * np.dot(canon_cloud - 0.5, R.T) + T
    posed_full_cloud  = S * np.dot(full - 0.5, R.T) + T
    num_plots = 3
    plt.figure(figsize=(6 * num_plots, 6))

    def plot(ax, pt_list, title):
        all_pts = np.concatenate(pt_list, axis=0)
        pmin, pmax = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2
        for pts in pt_list:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=5**2)
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
    cfg.item     ='nocs_synthetic'
    cfg.name_dset='nocs_synthetic'
    cfg.log_dir  = infos.second_path + cfg.log_dir
    dset = NOCSDataset(cfg=cfg, split='train')
    print('length', len(dset))
    for i in range(len(dset)):
        data = dset[i]
        print(f'--checking {i}th data')
        check_data(data)

if __name__ == '__main__':
    main()
    # python nocs_synthetic.py category='5' datasets=nocs_synthetic
