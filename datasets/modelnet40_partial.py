import numpy as np
import hydra
import random
import os
import glob
import scipy.io as sio
import torch
import pickle
import cv2
import torch.utils.data as data
from os.path import join as pjoin
import matplotlib.pyplot as plt
import __init__

try:
    import vgtk.so3conv.functional as L
    import vgtk.pc as pctk
except:
    pass
from datasets.modelnet40partial_render import backproject

def bp():
    import pdb;pdb.set_trace()

def rotation_distance_np(r0, r1):
    '''
    tip: r1 is usally the anchors
    '''
    if r0.ndim == 3:
        bidx = np.zeros(r0.shape[0]).astype(np.int32)
        traces = np.zeros([r0.shape[0], r1.shape[0]]).astype(np.int32)
        for bi in range(r0.shape[0]):
            diff_r = np.matmul(r1, r0[bi].T)
            traces[bi] = np.einsum('bii->b', diff_r)
            bidx[bi] = np.argmax(traces[bi])
        return traces, bidx
    else:
        # diff_r = np.matmul(r0, r1.T)
        # return np.einsum('ii', diff_r)

        diff_r = np.matmul(np.transpose(r1,(0,2,1)), r0)
        traces = np.einsum('bii->b', diff_r)

        return traces, np.argmax(traces), diff_r


def get_index(src_length, tgt_length):
    idx = np.arange(0, src_length)
    if src_length < tgt_length:
        idx = np.pad(idx, (0, tgt_length - src_length), 'wrap')
    idx = np.random.permutation(idx)[:tgt_length]
    return idx


def get_modelnet40_data(gt_path, meta, out_points):   # root_dset/render/airplane/train/0001/gt/001.npy
    """
    cloud_path = gt_path.replace('gt', 'cloud').replace('.npy', '.npz')   # precomputed partial cloud
    if os.path.exists(cloud_path):
        pts = np.load(cloud_path)['points']
    else:
    """
    depth_path = gt_path.replace('gt', 'depth').replace('.npy', '.png')
    depth = cv2.imread(depth_path, -1)
    pts = backproject(depth, meta['projection'], meta['near'], meta['far'],
                      from_image=True, vis=False)
    idx = get_index(len(pts), out_points)
    pts = pts[idx]
    gt_pose = np.load(gt_path)  # 4x4 matrix
    return pts, gt_pose


class ModelNet40Partial(data.Dataset):
    def __init__(self, cfg, mode=None):
        super(ModelNet40Partial, self).__init__()
        self.cfg = cfg
        self.mode = cfg.mode if mode is None else mode
        if 'val' in self.mode:
            self.mode = 'test'
        if cfg.use_fps_points:
            self.out_points = 4 * cfg.in_points
            print(f'---using {self.out_points} points as input')
        else:
            self.out_points = cfg.in_points

        self.add_noise = cfg.DATASET.add_noise
        self.noise_trans = cfg.DATASET.noise_trans

        self.dataset_path = cfg.DATASET.dataset_path
        self.render_path = pjoin(self.dataset_path, 'render', cfg.category, self.mode)
        self.points_path = pjoin(self.dataset_path, 'points', cfg.category, self.mode)

        with open(pjoin(self.render_path, 'meta.pkl'), 'rb') as f:
            self.meta_dict = pickle.load(f)  # near, far, projection
        self.instance_points, self.all_data = self.collect_data()
        try:
            self.anchors = L.get_anchors()
        except:
            self.anchors = np.random.rand(60, 3, 3)
        print(f"[Dataloader] : {self.mode} dataset size:", len(self.all_data))

    def collect_data(self):
        data_list = []
        instances = sorted([f for f in os.listdir(self.render_path) if os.path.isdir(pjoin(self.render_path, f))])
        instance_points = {}
        if self.cfg.DATASET.instance_ratio < 1 and self.mode == 'train':
            print('--original length is ', len(instances))
            random.shuffle(instances)
            instances = instances[:int(len(instances) * self.cfg.DATASET.instance_ratio)]
            print('--after length is ', len(instances))
        for instance in instances:
            cur_path = pjoin(self.render_path, instance, 'gt')
            items = [pjoin(cur_path, f) for f in os.listdir(cur_path) if f.endswith('.npy')]
            if self.cfg.DATASET.viewpoint_ratio < 1 and self.mode == 'train':
                print('--original length is ', len(items))
                random.shuffle(items)
                items = items[:int(len(items) * self.cfg.DATASET.viewpoint_ratio)]
                print('--after length is ', len(items))
            data_list += items
            points_path = pjoin(self.points_path, f'{instance}.npz')
            instance_points[instance] = np.load(points_path, allow_pickle=True)['points']

        return instance_points, data_list

    def get_complete_cloud(self, instance):
        pts = self.instance_points[instance]
        idx = get_index(len(pts), self.out_points)
        return pts[idx]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        category, _, instance = self.all_data[index].split('/')[-5:-2]  # ../render/airplane/train/0001/gt/001.npy
        model_points = self.get_complete_cloud(instance)
        boundary_pts = [np.min(model_points, axis=0), np.max(model_points, axis=0)]
        if 'ssl' not in self.cfg.task:
            center_pt = np.array([0, 0, 0]).astype(np.float32)
        else:
            center_pt = (boundary_pts[0] + boundary_pts[1])/2

        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        model_points = (model_points - center_pt.reshape(1, 3))/length_bb  + 0.5#

        cloud, gt_pose = get_modelnet40_data(self.all_data[index], self.meta_dict, self.out_points)
        cloud = cloud/length_bb
        target_r = gt_pose[:3, :3]
        target_t = gt_pose[:3, 3]/length_bb + center_pt.reshape(1, 3) @ target_r.T / length_bb
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        canon_cloud = np.dot(cloud - target_t, target_r) + 0.5
        if self.add_noise:
            cloud = cloud + add_t.astype(cloud.dtype)

        if self.cfg.augment:
            cloud, R = pctk.rotate_point_cloud(cloud)
            target_r = np.matmul(R, target_r)

        _, R_label, R0 = rotation_distance_np(target_r, self.anchors)

        R_gt = torch.from_numpy(target_r.astype(np.float32))  # predict r
        T = torch.from_numpy(target_t.reshape(1, 3).astype(np.float32))

        """
        target = np.dot(model_points, target_r.T)  # the complete point cloud corresponding to the observation
        if self.add_noise:
            target = np.add(target, target_t + add_t)  # include noise as well
        else:
            target = np.add(target, target_t)
        """
        if self.cfg.eval and self.cfg.pre_compute_delta:
            cloud = canon_cloud - 0.5
            R_gt  = torch.from_numpy(np.eye(3).astype(np.float32)) #
            T     = torch.from_numpy(np.zeros((1, 3)).astype(np.float32)) #

        data_dict = {
            'xyz': torch.from_numpy(cloud.astype(np.float32)),  # point cloud in camera space
            'points': torch.from_numpy(canon_cloud.astype(np.float32)),  # canonicalized xyz, in [0, 1]^3
            'full': torch.from_numpy(model_points.astype(np.float32)), # complete point cloud, in [0, 1]^3
            'label': torch.from_numpy(np.array([1]).astype(np.float32)),  # useless
            'R_gt': R_gt,
            'R_label': R_label,
            'R': torch.from_numpy(R0.astype(np.float32)),
            'T': T,
            'fn': self.all_data[index],
            'id': instance,
            'idx': index,
            'class': category
        }

        return data_dict


def check_data(data_dict):
    print(f'path: {data_dict["fn"]}, class: {data_dict["class"]}, instance: {data_dict["id"]}')
    cloud, canon_cloud, full = data_dict['xyz'], data_dict['points'], data_dict['full']
    R, T = data_dict['R_gt'].numpy(), data_dict['T'].numpy()
    posed_canon_cloud = np.dot(canon_cloud - 0.5, R.T) + T
    posed_full_cloud = np.dot(full - 0.5, R.T) + T

    num_plots = 3
    plt.figure(figsize=(6 * num_plots, 6))

    def plot(ax, pt_list, title):
        all_pts = np.concatenate(pt_list, axis=0)
        pmin, pmax = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2
        for pts in pt_list:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=2**2)

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
    dataset = ModelNet40Partial(cfg, 'test')
    print('length', len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        check_data(data)


if __name__ == '__main__':
    main()

"""
download modelnet_points.tar(complete object point clouds) and modelnet_render.tar (depth & gt pose)
modelnet40_partial/
    render/
         airplane/
            train/
                meta.pkl  # camera info
                0001/
                    depth/
                        000.png
                        001.png
                    gt/
                        000.npy
                        001.npy
    points/
        airplane/
            train/
                0001.npz
            test/


Pose and size:
- all models are of unit size
- the input data 'xyz' is NOT normalized -> still contains a big translation

Sampling:
- random sample from backprojected depth points

Caching:
- not implemented
"""
