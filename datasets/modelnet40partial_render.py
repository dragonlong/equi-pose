import trimesh
import numpy as np
import os
import multiprocessing as mp
import gc
import argparse
from tqdm import tqdm
import pyrender
import matplotlib.pyplot as plt
import cv2
from os.path import join as pjoin
import pickle

def bp():
    import pdb;pdb.set_trace()
def create_partial(read_path, save_folder, ins_num, render_num,
                   mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                   yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=False):
    m = trimesh.load(read_path)
    # centralization
    c = np.mean(m.vertices, axis=0)
    trans = np.eye(4)
    trans[:3, 3] = -c
    m.apply_transform(trans)
    scale = np.max(np.sqrt(np.sum(m.vertices ** 2, axis=1)))
    trans = np.eye(4)
    trans[:3, :3] = np.eye(3) / scale
    m.apply_transform(trans)

    scene = pyrender.Scene()

    mesh = pyrender.Mesh.from_trimesh(m)
    node = pyrender.Node(mesh=mesh, matrix=np.eye(4))

    scene.add_node(node)

    camera_pose = np.eye(4)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=pw / ph, znear=near, zfar=far)
    projection = camera.get_projection_matrix()
    scene.add(camera, camera_pose)
    r = pyrender.OffscreenRenderer(pw, ph)

    depth_path = pjoin(save_folder, ins_num, 'depth')
    os.makedirs(depth_path, exist_ok=True)
    gt_path = pjoin(save_folder, ins_num, 'gt')
    os.makedirs(gt_path, exist_ok=True)

    for i in range(render_num):
        pose = np.eye(4)
        pose[:3, 3] = mean_pose + np.random.randn(3) * std_pose
        rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        while upper_hemi and (rotation[1, 2] < 0 or rotation[2, 2] < 0):
            rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        pose[:3, :3] = rotation
        scene.set_pose(node, pose)
        depth_buffer = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)
        # pts = backproject(depth_buffer, projection, near, far, from_image=False)
        mask = depth_buffer > 0
        depth_z = buffer_depth_to_ndc(depth_buffer, near, far)  # [-1, 1]
        depth_image = depth_z * 0.5 + 0.5  # [0, 1]
        depth_image = linearize_img(depth_image, near, far)  # [0, 1]
        depth_image = np.uint16((depth_image * mask) * ((1 << 16) - 1))
        cv2.imwrite(pjoin(depth_path, f'{i:03}.png'), depth_image)
        np.save(pjoin(gt_path, f'{i:03}.npy'), pose)
        # backproject(depth_image, projection, near, far, from_image=True, vis=True)

    return projection, near, far


def proc_render(first, path_list, save_folder, render_num,
                mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=False):
    for read_path in tqdm(path_list):
        ins_num = read_path.split('/')[-1].split('.')[-2].split('_')[-1]
        projection, near, far = create_partial(read_path, save_folder, ins_num, render_num,
                                               mean_pose, std_pose,
                                               yfov, pw, ph, near, far, upper_hemi)
    if first:
        meta_path = pjoin(save_folder, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({'near': near, 'far': far, 'projection': projection}, f)


def ndc_depth_to_buffer(z, near, far):  # z in [-1, 1]
    return 2 * near * far / (near + far - z * (far - near))


def buffer_depth_to_ndc(d, near, far):  # d in (0, +
    return ((near + far) - 2 * near * far / np.clip(d, a_min=1e-6, a_max=1e6)) / (far - near)


def linearize_img(d, near, far):  # for visualization only
    return 2 * near / (near + far - d * (far - near))


def inv_linearize_img(d, near, far):  # for visualziation only
    return (near + far - 2 * near / d) / (far - near)


def backproject(depth, projection, near, far, from_image=False, vis=False):
    proj_inv = np.linalg.inv(projection)
    height, width = depth.shape
    non_zero_mask = (depth > 0)
    idxs = np.where(non_zero_mask)
    depth_selected = depth[idxs[0], idxs[1]].astype(np.float32).reshape((1, -1))
    if from_image:
        z = depth_selected / ((1 << 16) - 1)  # [0, 1]
        z = inv_linearize_img(z, near, far)  # [0, 1]
        z = z * 2 - 1.0  # [-1, 1]
        d = ndc_depth_to_buffer(z, near, far)
    else:
        d = depth_selected
        z = buffer_depth_to_ndc(d, near, far)

    grid = np.array([idxs[1] / width * 2 - 1, 1 - idxs[0] / height * 2])  # ndc [-1, 1]

    ones = np.ones_like(z)
    pts = np.concatenate((grid, z, ones), axis=0) * d  # before dividing by w, w = -z_world = d

    pts = proj_inv @ pts
    pts = np.transpose(pts)

    pts = pts[:, :3]

    if vis:
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=1)
        ax.set_xlim3d([center[0] - lim, center[0] + lim])
        ax.set_ylim3d([center[1] - lim, center[1] + lim])
        ax.set_zlim3d([center[2] - lim, center[2] + lim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    return pts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate_num', type=int, default=10)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--category', type=str, default='airplane')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_proc', type=int, default=8)
    parser.add_argument('--upper_hemi', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    read_folder = pjoin(args.input, args.category, args.split)
    save_folder = pjoin(args.output, args.category, args.split)
    path_list = [pjoin(read_folder, i) for i in os.listdir(read_folder) if i.endswith('off')]

    os.makedirs(save_folder, exist_ok=True)

    mean_pose_dict = {
        'airplane': np.array([0, 0, -1.8]),
        'car': np.array([0, 0, -2.0]),
        'bottle': np.array([0, 0, -2.0]),
        'bowl': np.array([0, 0, -2.3]),
        'sofa': np.array([0, 0, -2.3]),
        'chair': np.array([0, 0, -2.3])
    }
    mean_pose = mean_pose_dict[args.category]
    std_pose = np.array([0.2, 0.2, 0.15])

    mp.set_start_method('spawn')
    num_per_ins = (len(path_list) - 1) // args.num_proc + 1
    processes = []
    for i in range(args.num_proc):
        st = num_per_ins * i
        ed = min(st + num_per_ins, len(path_list))
        p = mp.Process(target=proc_render, args=(i == 0,
                                                 path_list[st: ed], save_folder, args.rotate_num,
                                                 mean_pose, std_pose,
                                                 np.deg2rad(60), 640, 480, 0.01, 10,
                                                 args.upper_hemi))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
