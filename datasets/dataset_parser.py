"""
light-weight loader class
"""
import numpy as np
import os
import sys
import random
import hydra
import torch
from omegaconf import OmegaConf
from os import makedirs, remove
from os.path import exists, join

import __init__
from global_info import global_info
from common.debugger import *
from datasets.parser import Parser
from datasets.modelnet40_complete import ModelNet40Complete # complete
from datasets.modelnet40_partial import ModelNet40Partial  # partial
from datasets.nocs_synthetic import NOCSDataset
from datasets.nocs_real import NOCSDatasetReal

from common.debugger import *
from common import vis_utils

epsilon = 10e-8
infos     = global_info()
second_path = infos.second_path
project_path= infos.project_path

def get_dataset(cfg,
                name_dset,
                split,
                train_it=True,
                use_cache=True):
    if name_dset == 'modelnet40_complete':
        print('using modelnet40 data ', split)
        return ModelNet40Complete(opt=cfg, mode=split)

    elif name_dset == 'modelnet40_partial':
        print('using modelnet40 data, new ', split)
        return ModelNet40Partial(cfg=cfg, mode=split)

    elif name_dset == 'nocs_synthetic':
        print('using nocs_synthetic data ', split)
        return NOCSDataset(cfg=cfg, root=cfg.DATASET.data_path, split=split)

    elif name_dset == 'nocs_real':
        if 'train' in split:
            split = 'real_train'
        else:
            split = 'real_test'
        print('using nocs_neweer data ', split_folder)
        return NOCSDatasetReal(cfg=cfg, root=cfg.DATASET.data_path, split=split_folder)

class DatasetParser(Parser):
    def __init__(self, cfg, mode='train', return_loader=True, domain=None, first_n=-1, add_noise=False, fixed_order=False, num_expr=0.01):
        name_dset  = cfg.name_dset
        print('name_dset', name_dset)
        collate = None
        self.train_dataset = get_dataset(
            cfg,
            name_dset,
            split='train',
            train_it=True)
        print("Final dataset size: {}".format(len(self.train_dataset)))

        self.valid_dataset = get_dataset(cfg,
            name_dset,
            split='val',
            train_it=False)
        print("Final dataset size: {}".format(len(self.valid_dataset)))

        drop_last = True  # Keeps batch_size constant
        if return_loader:
            self.trainloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=int(cfg.num_workers),
                pin_memory=True,
                drop_last=True,
            )
            self.validloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=cfg.test_batch,
                shuffle=False,
                collate_fn=collate,
                num_workers=int(cfg.num_workers),
                pin_memory=True,
                drop_last=True,
            )
        else:
            self.validloader = None
            self.trainloader = None
        self.testloader = None

@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(hydra.utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)
    parser = DatasetParser(cfg)
    #
    val_dataset   = parser.valid_dataset
    train_dataset = parser.train_dataset
    val_loader   = parser.validloader
    train_loader   = parser.trainloader

    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir
    dset          = train_dataset
    dloader       = train_loader


    if 'partial' in cfg.task:
        for j in range(100):
            if 'en3' in cfg.encoder_type:
                g_raw, g_real, n_arr, c_arr, m_arr, gt_points, instance_name, instance_name1, RR, center, idx, category_name = dset.__getitem__(j, verbose=True)
                RR = RR.cpu().numpy().reshape(-1, 3)
                center  = center.cpu().numpy()
                print(center)
                center  = center.mean(axis=0)
            else:
                g_raw, g_real, n_arr, c_arr, m_arr, gt_points, instance_name, instance_name1, RR, center_offset, idx, category_name = dset.__getitem__(j, verbose=True)
                RR = RR.cpu().numpy().reshape(-1, 3)
                center  = input - center_offset.cpu().numpy()
                print(center)
                center  = center.mean(axis=0)

            input = g_raw.ndata['x'].numpy()
            gt    = n_arr.transpose(1, 0).numpy()
            c_arr = c_arr.cpu().numpy()
            m_arr = m_arr.cpu().numpy().T
            full_pts = gt_points.transpose(1, 0).numpy()
            print(f'input: {input.shape}, gt: {gt.shape}')
            inds = [np.where(m_arr[:, 1]==0)[0], np.where(m_arr[:, 1]>0)[0]]
            if cfg.pred_6d:
                up_axis = np.matmul(np.array([[0.0, 1.0, 0.0]]), RR)
            else:
                up_axis = RR

            gt_vect= {'p': center, 'v': RR}
            # vis_utils.plot3d_pts([[input[inds[0]], input[inds[1]]]], [['hand', 'object']], s=2**2, arrows=[[gt_vect, gt_vect]], dpi=300, axis_off=False)
            # vis_utils.plot3d_pts([[input[inds[0]], input[inds[0]]], [gt]], [['input hand', 'hand'], ['gt NOCS']],  s=2**2, dpi=300, axis_off=False)
            # vis_utils.plot3d_pts([[input], [gt]], [['input'], ['gt NOCS']],  s=2**2, dpi=300, axis_off=False, color_channel=[[gt], [gt]])
            # vis_utils.plot3d_pts([[input], [full_pts]], [['input'], ['full shape']],  s=2**2, dpi=300, axis_off=False)
            # vis_utils.visualize_pointcloud([input, gt], title_name='partial + complete', backend='pyrender')
        else:
            indexs = np.random.randint(len(dset), size=100)
            for j in indexs:
                g_raw, n_arr, instance_name, RR, center_offset, idx, category_name = dset.__getitem__(j, verbose=True)
                gt_points = n_arr
                input = g_raw.ndata['x'].numpy()
                gt    = n_arr.numpy()
                full_pts = gt_points.numpy()
                print(f'input: {input.shape}, gt: {gt.shape}')
                vis_utils.plot3d_pts([[input], [gt]], [['input'], ['gt NOCS']],  s=2**2, dpi=300, axis_off=False, color_channel=[[gt], [gt]])
                vis_utils.plot3d_pts([[input], [full_pts]], [['input'], ['full shape']],  s=2**2, dpi=300, axis_off=False)


if __name__ == '__main__':
    main()
