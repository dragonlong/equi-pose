import numpy as np
import trimesh
import os
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
import __init__
import vgtk.pc as pctk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np
import random
def bp():
    import pdb;pdb.set_trace()

class ModelNet40Complete(data.Dataset):
    def __init__(self, opt, mode=None):
        super(ModelNet40Complete, self).__init__()
        self.opt = opt

        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        # attention method: 'attention | rotation'
        self.flag = opt.model.flag

        self.anchors = L.get_anchors()

        if opt.category:
            cats = [opt.category]
            print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")
        else:
            cats = os.listdir(opt.DATASET.dataset_path)
        if 'val' in self.mode:
            self.mode = 'test'
        self.dataset_path = opt.DATASET.dataset_path
        self.all_data = []
        for cat in cats:
            for fn in sorted(glob.glob(os.path.join(opt.DATASET.dataset_path, cat, self.mode, "*.mat"))):
                self.all_data.append(fn)
        if opt.DATASET.instance_ratio < 1 and 'train' in self.mode:
            random.shuffle(self.all_data)
            self.all_data = self.all_data[:int(len(self.all_data) * opt.DATASET.instance_ratio)]
            print(f'We end up with {opt.DATASET.instance_ratio} of the original data')
        print("[Dataloader] : Training dataset size:", len(self.all_data))

        if self.opt.no_augmentation:
            print("[Dataloader]: USING ALIGNED MODELNET LOADER!")
        else:
            print("[Dataloader]: USING ROTATED MODELNET LOADER!")

        if self.opt.eval:
            np.random.seed(seed=233423)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = sio.loadmat(self.all_data[index])
        if self.opt.use_fps_points:
            _, pc = pctk.uniform_resample_np(data['pc'], 4 * self.opt.in_points)
        else:
            _, pc = pctk.uniform_resample_np(data['pc'], self.opt.in_points)

        boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # all normalize into 0
        pc_canon = (pc - center_pt.reshape(1, 3))/length_bb
        pc = np.copy(pc_canon)   # centered at 0
        pc_canon = pc_canon + 0.5 # NOCS space

        R = np.eye(3)
        R_label = 29
        t = np.random.rand(1, 3)
        T = torch.from_numpy(t.astype(np.float32))
        if self.opt.augment and not self.opt.pre_compute_delta:
            if 'R' in data.keys() and self.mode != 'train':
                pc, R = pctk.rotate_point_cloud(pc, data['R'])
            else:
                pc, R = pctk.rotate_point_cloud(pc)
            R_gt = np.copy(R)
        else:
            R_gt = np.copy(R)
        _, R_label, R0 = rotation_distance_np(R, self.anchors)
        if self.opt.pred_t:
            pc = pc + t
        else:
            T = T * 0
        num_index = self.all_data[index].split('/')[-1].split('.')[0].split('_')[-1]
        return {'xyz': torch.from_numpy(pc.astype(np.float32)),
                'points': torch.from_numpy(pc_canon.astype(np.float32)),
                'label': torch.from_numpy(data['label'].flatten()).long(),
                'R_gt' : torch.from_numpy(R_gt.astype(np.float32)),
                'R_label': torch.Tensor([R_label]).long(),
                'R': R0,
                'T': T,
                'fn': data['name'][0],
                'id': num_index,
                'idx': num_index,
                'class': self.all_data[index].split('/')[-3]
               }

if __name__ == '__main__':
    from models.spconv.options import opt
    BS = 2
    N  = 1024
    C  = 3
    device = torch.device("cuda:0")
    x = torch.randn(BS, N, 3).to(device)
    opt.model.model = 'inv_so3net'
    opt.category = 'airplane'
    dset = ModelNet40Complete(opt, mode='test')
    bp()
    for i in range(10):
        dp = dset.__getitem__(i)
        print(dp)
    print('Con!')
