import torch
import torch.nn as nn
import hydra
import numpy as np
from sklearn.decomposition import PCA

import __init__
from global_info import global_info
epsilon = 10e-8

def breakpoint():
    import pdb;pdb.set_trace()

class Parser():
    def __init__(self, cfg, Dataset=None, instantiate=True): # with cfg and Dataset
        self.workers       = cfg.num_workers # workers is free to decide
        self.batch_size    = cfg.TRAIN.batch_size
        self.shuffle_train = cfg.TRAIN.shuffle_train

        if instantiate:
            self.train_dataset = Dataset(
                cfg=cfg,
                add_noise=cfg.TRAIN.train_data_add_noise,
                first_n=cfg.TRAIN.train_first_n,
                mode='train',
                fixed_order=False,)

            self.train_sampler = None
            self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle=(self.shuffle_train and self.train_sampler is None),
                                                         num_workers=self.workers,
                                                         pin_memory=True,
                                                         drop_last=True)
            assert len(self.trainloader) > 0

            # seen instances
            self.valid_dataset = Dataset(
                cfg,
                add_noise=cfg.TRAIN.val_data_add_noise,
                first_n=cfg.TRAIN.val_first_n,
                mode='test',
                domain='seen',
                fixed_order=True,
                )

            self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=False,
                                                           num_workers=self.workers,
                                                           pin_memory=True,
                                                           drop_last=True)
            assert len(self.validloader) > 0

            # unseen instances
            self.test_dataset = Dataset(
                cfg,
                add_noise=cfg.TRAIN.val_data_add_noise,
                first_n=cfg.TRAIN.val_first_n,
                mode='test',
                domain='unseen',
                fixed_order=True,
                )

            self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.workers,
                                                        pin_memory=True,
                                                        drop_last=True)
            assert len(self.testloader) > 0

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)


import os
import random
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
@hydra.main(config_path="../config/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    infos       = global_info()
    second_path       = infos.second_path
    group_dir       = infos.group_path
    second_path     = infos.second_path

    # category-wise training setup
    cfg.is_test = False

    random.seed(30)
    parser = Parser(cfg)
    train_loader   = parser.get_train_set()
    valid_loader   = parser.get_valid_set()
    hand_joints_list = []

    # points
    for i, batch in enumerate(train_loader):
        print('iterate', i)
        if i > 1000:
            break
        hand_joints_cam_centered = batch['hand_joints_gt'].cpu().numpy().astype(np.float32)
        hand_joints_list.append(hand_joints_cam_centered)
    all_hands = np.concatenate(hand_joints_list, axis=0)

    pca = PCA(n_components=30)
    principalComponents = pca.fit_transform(all_hands)
    save_dict = {}
    save_dict['all_hands'] = all_hands
    save_dict['E'] = pca.components_
    save_dict['S'] = pca.singular_values_
    save_dict['C'] = principalComponents
    save_dict['M'] = pca.mean_

    save_name = second_path + f'/data/pickle/hand_pca.npy'
    np.save(save_name, save_dict)
    print('saving to ', save_name)
    print('PCA has shape: ', principalComponents.shape)


if __name__ == '__main__':
    main()
