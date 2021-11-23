import os
import time
import random
import numpy as np
from os import makedirs, remove
from os.path import exists, join
from numpy.linalg import matrix_rank, inv
from plyfile import PlyData, PlyElement
from omegaconf import DictConfig, ListConfig

import torch
import operator as op
from functools import reduce
from copy import deepcopy

def boolean_string(s):
    if s is None:
        return None
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def update_dict(old_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict):
            if not key in old_dict:
                old_dict[key] = {}
            update_dict(old_dict[key], value)
        else:
            old_dict[key] = value


def merge_dict(old_dict, new_dict, init_val=[], f=lambda x, y: x + [y]):
    for key, value in new_dict.items():
        if not key in old_dict.keys():
            old_dict[key] = init_val
        old_dict[key] = f(old_dict[key], value)


def detach_dict(d):
    for key, value in d.items():
        if isinstance(value, dict):
            detach_dict(value)
        elif isinstance(value, torch.Tensor):
            d[key] = value.detach().cpu().numpy()


def add_dict(old_dict, new_dict):
    """
    for key, value in new_dict.items():
        if isinstance(value, dict):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().numpy()
        if not key in old_dict:
            old_dict[key] = value
        else:
            old_dict[key] += value
    return None
    """

    def copy_dict(d):
        ret = {}
        for key, value in d.items():
            if isinstance(value, dict):
                ret[key] = copy_dict(value)
            else:
                ret[key] = value
        del d
        return ret
    detach_dict(new_dict)
    for key, value in new_dict.items():
        if not key in old_dict.keys():
            if isinstance(value, dict):
                old_dict[key] = copy_dict(value)
            else:
                old_dict[key] = value
        else:
            if isinstance(value, dict):
                add_dict(old_dict[key], value)
            else:
                old_dict[key] += value


def ensure_dir(path, verbose=False):
    if not os.path.exists(path):
        if verbose:
            print("Create folder ", path)
        os.makedirs(path)
    else:
        if verbose:
            print(path, " already exists.")


def ensure_dirs(paths):
    if isinstance(paths, list):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def write_loss(it, loss_dict, writer):
    def write_dict(d, prefix=None):
        for key, value in d.items():
            name = str(key) if prefix is None else '/'.join([prefix, str(key)])
            if isinstance(value, dict):
                write_dict(value, name)
            else:
                writer.add_scalar(name, value, it + 1)
    write_dict(loss_dict)


def log_loss_summary(loss_dict, cnt, log_loss):
    def log_dict(d, prefix=None):
        for key, value in d.items():
            name = str(key) if prefix is None else '/'.join([prefix, str(key)])
            if isinstance(value, dict):
                log_dict(value, name)
            else:
                log_loss(name, d[key] / cnt)
    log_dict(loss_dict)


def divide_dict(ddd, cnt):
    def div_dict(d):
        ret = {}
        for key, value in d.items():
            if isinstance(value, dict):
                ret[key] = div_dict(value)
            else:
                ret[key] = value / cnt
        return ret
    return div_dict(ddd)


def print_composite(data, beg="", depth=1000):
    if depth <= 0:
        print(f'{beg} ...')
        return
    if isinstance(data, dict):
        print(f'{beg} dict, size = {len(data)}')
        for key, value in data.items():
            print(f'  {beg}{key}:')
            print_composite(value, beg + "    ", depth - 1)
    elif isinstance(data, list):
        print(f'{beg} list, len = {len(data)}')
        for i, item in enumerate(data):
            print(f'  {beg}item {i}')
            print_composite(item, beg + "    ", depth - 1)
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        print(f'{beg} array of size {data.shape}')
    else:
        print(f'{beg} {data}')


class Timer:
    def __init__(self, on):
        self.on = on
        self.cur = time.time()

    def tick(self, str=None):
        if not self.on:
            return
        cur = time.time()
        diff = cur - self.cur
        self.cur = cur
        if str is not None:
            print(str, diff)
        return diff


def get_ith_from_batch(data, i, to_single=True):
    if isinstance(data, dict):
        return {key: get_ith_from_batch(value, i, to_single) for key, value in data.items()}
    elif isinstance(data, list):
        return [get_ith_from_batch(item, i, to_single) for item in data]
    elif isinstance(data, torch.Tensor):
        if to_single:
            return data[i].detach().cpu().item()
        else:
            return data[i].detach().cpu()
    elif isinstance(data, np.ndarray):
        return data[i]
    elif data is None:
        return None
    elif isinstance(data, str):
        return data
    else:
        assert 0, f'Unsupported data type {type(data)}'


def cvt_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.tensor(x).float().to(device)
    elif isinstance(x, torch.Tensor):
        return x.float().to(device)
    elif isinstance(x, dict):
        return {key: cvt_torch(value, device) for key, value in x.items()}
    elif isinstance(x, list):
        return [cvt_torch(item, device) for item in x]
    elif x is None:
        return None


class Mixture:
    def __init__(self, proportion_dict):
        self.keys = list(proportion_dict.keys())
        self.cumsum = np.cumsum([proportion_dict[key] for key in self.keys])
        assert self.cumsum[-1] == 1.0, 'Proportions do not sum to one'

    def sample(self):
        choice = random.random()
        idx = np.searchsorted(self.cumsum, choice)
        return self.keys[idx]


def inspect_tensors(verbose=False):
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tmp = reduce(op.mul, obj.size())
                total += tmp
                if verbose and obj.size() == (1, 128, 128) or obj.size() == (1, 256, 256) or obj.size() == (1, 128, 256) or obj.size() == (1, 256, 128):
                    print(obj.size(), obj)
                    # print(type(obj), obj. tmp, obj.size())
        except:
            pass
    print("=================== Total = {} ====================".format(total))


def eyes_like(tensor: torch.Tensor):  # [Bs, 3, 3]
    assert tensor.shape[-2:] == (3, 3), 'eyes must be applied to tensor w/ last two dims = (3, 3)'
    eyes = torch.eye(3, dtype=tensor.dtype, device=tensor.device)
    eyes = eyes.reshape(tuple(1 for _ in range(len(tensor.shape) - 2)) + (3, 3)).repeat(tensor.shape[:-2] + (1, 1))
    return eyes


def flatten_dict(loss_dict):
    def flatten_d(d, prefix=None):
        new_d = {}
        for key, value in d.items():
            name = str(key) if prefix is None else '/'.join([prefix, str(key)])
            if isinstance(value, dict):
                new_d.update(flatten_d(value, name))
            else:
                new_d[name] = value
        return new_d
    ret = flatten_d(loss_dict)
    return ret


def per_dict_to_csv(loss_dict, csv_name):
    all_ins = {inst: flatten_dict(loss_dict[inst]) for inst in loss_dict}

    keys = list(list(all_ins.values())[0].keys())
    dir = os.path.dirname(csv_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(csv_name, 'w') as f:
        def fprint(s):
            print(s, end='', file=f)
        for key in keys:
            fprint(f',{key}')
        fprint('\n')
        for inst in all_ins:
            fprint(f'{inst}')
            for key in keys:
                fprint(f',{all_ins[inst][key]}')
            fprint('\n')


COLOR_MAP_RGB =  [[0, 0, 0],
                [0, 0, 255],
                [245, 150, 100],
                [245, 230, 100],
                [250, 80, 100],
                [150, 60, 30],
                [255, 0, 0],
                [180, 30, 80],
                [255, 0, 0],
                [30, 30, 255],
                [200, 40, 255],
                [90, 30, 150],
                [255, 0, 255],
                [255, 150, 255],
                [75, 0, 75],
                [75, 0, 175],
                [0, 200, 255],
                [50, 120, 255],
                [0, 150, 255],
                [170, 255, 150],
                [0, 175, 0],
                [0, 60, 135],
                [80, 240, 150],
                [150, 240, 255],
                [0, 0, 255],
                [255, 255, 50],
                [245, 150, 100],
                [255, 0, 0],
                [200, 40, 255],
                [30, 30, 255],
                [90, 30, 150],
                [250, 80, 100],
                [180, 30, 80],
                [255, 0, 0]]
#
def breakpoint():
    import pdb; pdb.set_trace()

def bp():
    import pdb;pdb.set_trace()
def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)

def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)

def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)

def dist_tester():
    print('please use ')

def check_path(save_path):
    # save_path = self.cfg.log_dir + '/input'
    if not exists(save_path):
        print('not exist, creating the path')
        makedirs(save_path)

def summary(features):
    if isinstance(features, dict):
        for k, v in features.items():
            print(f'type: {k}; Size: {v.size()}')
    else:
        print(f'Size: {features.size()}')
        print(features[0])

def print_group(values, names=None):
    if names is None:
        for j in range(len(values)):
            print(values[j], '\n')
    else:
        for j in range(len(names)):
            print(names[j], '\n', values[j], '\n')

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def colorize_pointcloud(xyz, label, ignore_label=255):
  assert label[label != ignore_label].max() < len(COLOR_MAP_RGB), 'Not enough colors.'
  label_rgb = np.array([COLOR_MAP_RGB[i] if i != ignore_label else IGNORE_COLOR for i in label])
  return np.hstack((xyz, label_rgb))


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7
    python_types = (float, float, float, int, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1'), ('label', 'u1')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary is True:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    # PlyData([el], text=True).write(filename)
    with open(filename, 'w') as f:
      f.write('ply\n'
              'format ascii 1.0\n'
              'element vertex %d\n'
              'property float x\n'
              'property float y\n'
              'property float z\n'
              'property uchar red\n'
              'property uchar green\n'
              'property uchar blue\n'
              'property uchar alpha\n'
              'end_header\n' % points_3d.shape[0])
      for row_idx in range(points_3d.shape[0]):
        X, Y, Z, R, G, B = points_3d[row_idx]
        f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
  if verbose is True:
    print('Saved point cloud to: %s' % filename)

def check_h5_keys(hf, verbose=False):
    print('')
    if verbose:
        for key in list(hf.keys()):
            print(key, hf[key][()].shape)

def save_for_viz(names, objects, save_name='default.npy', type='tensor', verbose=True):
    # names = ['coords', 'input', 'target']
    # objects = [coords, input, target]
    save_input_data = True
    print('---saving to ', save_name)
    if save_input_data:
        viz_dict = {}
        for name, item in zip(names, objects):
            if type == 'tensor':
                viz_dict[name] = item.cpu().numpy()
            else:
                viz_dict[name] = item
            if verbose:
                try:
                    print(name, ': ', item.shape)
                except:
                    print(name, ': ', item)
        # save_name = f'{self.config.save_viz_dir}/input_{iteration}.npy'
        save_path = os.path.dirname(save_name)
        if not exists(save_path):
            makedirs(save_path)
        if len(objects) == 2 and 'labels' in names and 'points' in names:
            xyzs = colorize_pointcloud(viz_dict['points'][:, :3], viz_dict['labels'].astype(np.int8))
            save_point_cloud(xyzs, save_name.replace('npy', 'ply'), with_label=False)
        else:
            np.save(save_name, arr=viz_dict)

def visualize_results(coords, input, target, upsampled_pred, config, iteration):
  # Get filter for valid predictions in the first batch.
  target_batch = coords[:, 3].numpy() == 0
  input_xyz = coords[:, :3].numpy()
  target_valid = target.numpy() != 255
  target_pred = np.logical_and(target_batch, target_valid)
  target_nonpred = np.logical_and(target_batch, ~target_valid)
  ptc_nonpred = np.hstack((input_xyz[target_nonpred], np.zeros((np.sum(target_nonpred), 3))))
  # Unwrap file index if tested with rotation.
  file_iter = iteration
  if config.test_rotation >= 1:
    file_iter = iteration // config.test_rotation
  # Create directory to save visualization results.
  os.makedirs(config.visualize_path, exist_ok=True)
  # Label visualization in RGB.
  xyzlabel = colorize_pointcloud(input_xyz[target_pred], upsampled_pred[target_pred])
  xyzlabel = np.vstack((xyzlabel, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'pred', '%04d.ply' % file_iter])
  save_point_cloud(xyzlabel, os.path.join(config.visualize_path, filename), verbose=False)
  # RGB input values visualization.
  xyzrgb = np.hstack((input_xyz[target_batch], input[:, :3].cpu().numpy()[target_batch]))
  filename = '_'.join([config.dataset, config.model, 'rgb', '%04d.ply' % file_iter])
  save_point_cloud(xyzrgb, os.path.join(config.visualize_path, filename), verbose=False)
  # Ground-truth visualization in RGB.
  xyzgt = colorize_pointcloud(input_xyz[target_pred], target.numpy()[target_pred])
  xyzgt = np.vstack((xyzgt, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'gt', '%04d.ply' % file_iter])
  save_point_cloud(xyzgt, os.path.join(config.visualize_path, filename), verbose=False)
