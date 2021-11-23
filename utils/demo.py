import pyvista as pv
from pyvista import examples
import numpy as np
from os import makedirs, remove
from os.path import exists, join
from glob import glob
import scipy.io as sio
import argparse

def bp():
    import pdb;pdb.set_trace()

import __init__
from global_info import global_info
infos           = global_info()

project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories
my_dir          = infos.second_path
delta_R         = infos.delta_R
delta_T         = infos.delta_T

def get_axis(r_mat):
    x_axis = np.matmul(np.array([[1.0, 0.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    y_axis = np.matmul(np.array([[0.0, 1.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    z_axis = np.matmul(np.array([[0.0, 0.0, 1.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    return x_axis, y_axis, z_axis

class simple_config(object):
    def __init__(self, target_entry=None, exp_num=None, category=None, name_dset=None, icp_method_type=0):
        self.canonical_path=f'{my_dir}/data/modelnet40_complete/EvenAlignedModelNet40PC'
        if target_entry is None:
            self.exp_num    = exp_num
            self.category = category
            self.name_dset  = name_dset
            self.symmetry_type = 0
        else:
            self.category=target_entry.split('_')[1]

            if target_entry.split('_')[0] == 'complete':
                self.name_dset='modelnet40_complete'
            else:
                self.name_dset='modelnet40_partial'
            if self.name_dset == 'modelnet40_complete':
                if self.category == 'airplane':
                    self.exp_num    = '0.813'
                elif self.category == 'car':
                    self.exp_num    = '0.851'
                elif self.category == 'chair':
                    self.exp_num    = '0.8581'
                elif self.category == 'sofa':
                    self.exp_num    = '0.8591'
                elif self.category == 'bottle':
                    self.exp_num    = '0.8562'
                    self.symmetry_type = 1
                    self.chosen_axis = 'z'
            elif self.name_dset == 'modelnet40_partial':
                # self.canonical_path=f'{my_dir}/data/modelnet40_partial/render/{category}/test/gt'
                if self.category == 'airplane':
                    self.exp_num    = '0.913r'
                elif self.category == 'car':
                    self.exp_num    = '0.921r'
                elif self.category == 'chair':
                    self.exp_num    = '0.951r'
                elif self.category == 'sofa':
                    self.exp_num    = '0.961r'
                elif self.category == 'bottle':
                    self.exp_num    = '0.941r'
                    self.symmetry_type = 1
                    self.chosen_axis = 'z'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_entry', default='complete_airplane', help='yyds')
    args = parser.parse_args()
    np.random.seed(0)
    pv.set_plot_theme("document")
    off_screen = False
    color      = 'gold' #'deepskyblue'
    k = 1.3
    k1 = 1.05
    point_size   = 10
    font_size    = 18
    num  = 100
    colors = np.random.rand(num, 3)
    colors_pred = np.random.rand(num, 3)
    colors_axes = ['r', 'g', 'b']

    target_entry    = args.target_entry # 'complete_airplane' # '0.961r' #'0.94r' # 0.92 #0.81
    cfg = simple_config(target_entry)
    query_keys   = ['canon', 'target', 'input', 'pred'] #
    fpath   = f'/home/dragonx/Dropbox/neurips21/results/preds/{cfg.exp_num}/generation/' # generation/generation
    gt_path = '/home/dragonx/Dropbox/neurips21/results'
    fnames = {}
    for key in query_keys:
        fnames[key] = sorted(glob(f'{fpath}/*_0*_{key}*txt'))
        print(key, ': ', len(fnames[key]))
    idxs = np.random.permutation(len(fnames['canon']))
    idxs = idxs[1:]
    # print(idxs)

    # >>>>>    set delta_R/T
    entry_key = f'{cfg.exp_num}_{cfg.name_dset}_{cfg.category}'
    delta_r = np.squeeze(delta_R[entry_key]) # I --> delta_r,
    if entry_key in delta_T:
        delta_t = delta_T[entry_key]
    else:
        delta_t = None
    # >>>>> set gt pose annotations
    # ['basename', 'in', 'r_raw', 'r_gt', 't_gt', 's_gt', 'r_pred', 't_pred', 's_pred']
    gt_annos = np.load(f'{gt_path}/{cfg.name_dset}/{cfg.exp_num}_unseen_part_rt_pn_general.npy', allow_pickle=True).item()
    if 'partial' in args.target_entry:
        gt_annos['info']['basename'].reverse()
        gt_annos['info']['r_gt'].reverse()
        gt_annos['info']['t_gt'].reverse()
        gt_annos['info']['s_gt'].reverse()
        gt_annos['info']['r_pred'].reverse()
        gt_annos['info']['t_pred'].reverse()
        # gt_annos['info']['s_pred'] = gt_annos['info']['s_pred'].reverse()
    #>>>>>>    set plotter
    total_num = min(16, len(idxs) - len(idxs) % num)
    col_num = int(np.sqrt(total_num))
    row_num = int(total_num/col_num)
    p = pv.Plotter(notebook=0, shape=(1, 3), border=False)
    p.open_gif(f"./viz/{entry_key}.gif")
    dx = 1.2
    dy = 1.2
    # >>>>>>>>>>>>    canonical reconstruction, GT points
    keys = ['input', 'canon', 'pred'] # 'target',
    titles = ['Input of Unseen Instances(partial) ', 'Canonical Shape Prediction', '6D Pose Prediction']
    m = 0
    for n in range(3):
        p.subplot(m,n)
        p.add_title(titles[n], font_size=font_size)
        for i in range(row_num):
            for j in range(col_num):
                index = idxs[i*col_num + j]
                fn = fnames['canon'][index].replace('canon', keys[n])
                instance_id = fn.split('.txt')[0].split('/')[-1].split('_')[2]
                if keys[n] == 'input':
                    point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
                elif keys[n] == 'canon':
                    if delta_t is not None:
                        point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5 - delta_t.reshape(1, 3) ) @ delta_r.T + 0.5)
                    else:
                        point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5) @ delta_r.T + 0.5)
                elif keys[n] == 'pred':
                    basename = f'0_{instance_id}_{cfg.category}'
                    index_info = gt_annos['info']['basename'].index(basename)
                    gt_r = np.array(gt_annos['info']['r_pred'][index_info])
                    gt_t = gt_annos['info']['t_pred'][index_info]
                    # gt_s = gt_annos['info']['s_pred'][index_info]
                    fn = fnames['canon'][index].replace('canon', 'input')
                    point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
                    cent = np.array([[dx*i, dy*j, 0]]).repeat(3, axis=0) + np.array(gt_t).reshape(1, 3)
                    direction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ gt_r.T
                    for k in range(3):
                        p.add_arrows(cent[k:k+1], direction[k:k+1], color=colors_axes[k], line_width=2, opacity=0.5, mag=0.8)

                elif keys[n] == 'target':
                    complete_fname = cfg.canonical_path + f'/{cfg.category}/test/{cfg.category}_{instance_id}.mat'
                    pc = sio.loadmat(complete_fname)['pc']
                    boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
                    center_pt = (boundary_pts[0] + boundary_pts[1])/2
                    length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
                    pc_canon = (pc - center_pt.reshape(1, 3))/length_bb
                    pc_canon = pc_canon + 0.5 # NOCS space
                    pc_canon = np.random.permutation(pc_canon)[:3072]
                    point_cloud = pv.PolyData(pc_canon.astype(np.float32)[:, :3])
                point_cloud.points = point_cloud.points + np.array([[dx*i, dy*j, 0]])
                # add mesh one by one
                p.add_mesh(point_cloud, color=colors[i*col_num +j], point_size=point_size, render_points_as_spheres=True)
                p.camera_position = [(14.92649241420315, 13.342057943666862, 12.463744221100175),
                                     (-0.028118640184402466, 0.008874431252479553, 0.02914293110370636),
                                     (-0.37331596991022065, -0.37332276444831175, 0.8492733954120201)]
                if 'partial' in args.target_entry:
                    p.camera_position = [(13.730788290905835, 11.435420241787062, 15.396641572920528),
                                     (-0.028118640184402466, 0.008874431252479553, 0.02914293110370636),
                                     (-0.4641430273776987, -0.45952287562631616, 0.757238388430477)]
                if i== 0 and j == 0:
                    p.show(auto_close=False)
                print(p.camera_position)
                print('writing frames')
                p.add_floor()
                p.write_frame()
                p.write_frame()
                p.write_frame()
                # p.clear()
    p.link_views()  # link all the views
    #
    p.show(auto_close=False)
    print('writing frames')
    p.write_frame()
    p.close()
    #
    # if cfg.name_dset == 'modelnet40_partial':
    #     p = pv.Plotter(notebook=0, shape=(2, 10), border=False)
    #
    #     cfg = simple_config(target_entry)
    #     # get delta_r, delta_t
    #     entry_key = f'{cfg.exp_num}_{cfg.name_dset}_{cfg.category}'
    #     delta_r = np.squeeze(delta_R[entry_key]) # I --> delta_r,
    #     if entry_key in delta_T:
    #         delta_t = delta_T[entry_key]
    #     else:
    #         delta_t = None
    #     for i in range(num):
    #         # p.subplot(0,i)
    #         index = idxs[i]
    #         # fn = fnames['canon'][index]
    #         # if delta_t is not None:
    #         #     point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5 - delta_t.reshape(1, 3) ) @ delta_r.T + 0.5)
    #         # else:
    #         #     point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5) @ delta_r.T + 0.5)
    #         # p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
    #         # p.subplot(1,i)
    #         # instance_id = fn.split('.txt')[0].split('/')[-1].split('_')[2]
    #         # complete_fname = cfg.canonical_path + f'/{cfg.category}/test/{cfg.category}_{instance_id}.mat'
    #         #
    #         # pc = sio.loadmat(complete_fname)['pc']
    #         # boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
    #         # center_pt = (boundary_pts[0] + boundary_pts[1])/2
    #         # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
    #         # pc_canon = (pc - center_pt.reshape(1, 3))/length_bb
    #         # pc_canon = pc_canon + 0.5 # NOCS space
    #         # pc_canon = np.random.permutation(pc_canon)[:3072]
    #         # point_cloud = pv.PolyData(pc_canon.astype(np.float32)[:, :3])
    #         # # complete_fname = glob(fpath.replace('partial', 'complete') + f'*{instance_id}_target*')
    #         # # point_cloud = pv.PolyData(np.loadtxt(complete_fname[0], delimiter=' ').astype(np.float32)[:, :3])
    #         # p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
    #
    #         p.subplot(0,i)
    #         fn = fnames['canon'][index].replace('canon', 'input')
    #         point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
    #         p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
    #
    #         p.subplot(1,i)
    #         p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
    #         fn = fnames['canon'][index].replace('canon', 'pred')
    #         point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
    #         p.add_mesh(point_cloud, color=colors_pred[i], point_size=point_size, render_points_as_spheres=True)
    #
    #     p.link_views()  # link all the views
    #     p.camera_position = [(0.5, 0.5, 5), (0.5, 0.5, 0.5), (-1, 0, 0)]
    #
    #     p.show(auto_close=False)
    #     print(p.camera_position)
    #     p.open_gif(f"./viz/{entry_key}.gif")
    #
    #     nframe = 1
    #     for i in range(nframe):
    #         p.camera_position = [(0.5, 0.5, 5), (0.5, 0.5, 0.5), (-1, 0, 0)]
    #         p.write_frame()
    #
    #     p.close()
    # # p = pv.Plotter(off_screen=off_screen, lighting='light_kit')
    # # for i in idxs:
    # #     fn = fnames['canon'][i].replace('canon', 'target')
    # #     point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3]+0.4*i)
    # #     p.add_mesh(point_cloud, color=np.random.rand(3), point_size=5, render_points_as_spheres=True)
    # # p.add_title('human-aligned shape space(100 unseen)', font_size=font_size)
    # # p.show()
    #
    # # for fn in fnames['canon']:
    # #     points = {}
    # #     for key in query_keys:
    # #         point_normal_set = np.loadtxt(fn.replace('canon', key), delimiter=' ').astype(np.float32)
    # #         pts = point_normal_set[:, :3]
    # #         # if key == 'input':
    # #         #     refer_shift = pts.mean(axis=0, keepdims=True)
    # #         # if key == 'pred':
    # #         #     pts = pts + refer_shift
    # #         # r_mat  = point_normal_set[:, 4:].reshape(-1, 3, 3)
    # #         points[key] = pv.PolyData(pts)
    # #     # point_cloud['vectors'] = x_axis[:, 0, :]
    # #     # arrows = point_cloud.glyph(orient='vectors', scale=False, factor=0.15,)
    # #     p = pv.Plotter(off_screen=off_screen, lighting='light_kit', shape=window_shape)
    # #     p.add_mesh(points['target'], color='r', point_size=15, render_points_as_spheres=True)
    # #     # p.camera_position = [(0.1*k, 1.8*k, 2.0*k),
    # #     #                  (0.5, 0.5, 0.5),
    # #     #                  (-0.9814055156571295, -0.1437895877734097, -0.12715253943872534)]
    # #     p.camera_position = [(0.42189609206584944, 0.3720949834155453, 3.312479348599398),
    # #                      (0.5, 0.5, 0.5),
    # #                      (-0.999559884631558, 0.011789591923156638, -0.027222096863255326)]
    # #     # p.add_legend([['nocs', 'r']], bcolor=(1.0, 1.0, 1.0))
    # #     p.add_title('input(canonicalized)', font_size=font_size)
    # #     p.show_grid()
    # #
    # #     p.subplot(0,1)
    # #     p.add_mesh(points['canon'], color='g', point_size=15, render_points_as_spheres=True)
    # #     # p.camera_position = [(0.1*k, 1.8*k, 2.0*k),
    # #     #                  (0.5, 0.5, 0.5),
    # #     #                  (-0.9814055156571295, -0.1437895877734097, -0.12715253943872534)]
    # #     p.camera_position = [(1.6224649540075546*k1, 2.558959462540017*k1, -0.18487386521674765*k1),
    # #                      (0.5058576315641403, 0.5140270888805389, 0.5149073377251625),
    # #                      (-0.1485912819656584, -0.24657874854598089, -0.9576635900405216)]
    # #
    # #     # p.add_legend([['pred', 'g']], bcolor=(1.0, 1.0, 1.0))
    # #     p.add_title('predicted shape', font_size=font_size)
    # #     p.show_grid()
    # #     p.subplot(0,2)
    # #     sphere = pv.Sphere(radius=0.1)
    # #     p.add_mesh(sphere,  color='b')
    # #     p.add_mesh(points['pred'],  color='g', point_size=15, render_points_as_spheres=True)
    # #     p.add_mesh(points['input'], color='r', point_size=15, render_points_as_spheres=True)
    # #     p.add_legend([['pred', 'g'], ['input', 'r']], bcolor=(1.0, 1.0, 1.0))
    # #     p.add_title('pose estimation', font_size=font_size)
    # #     p.show_grid()
    # #
    # #     # p.subplot(0,3)
    # #     # sphere = pv.Sphere(radius=0.1)
    # #     # p.add_mesh(sphere,  color='b')
    # #     #
    # #     # p.add_mesh(points['icp'],  color='g', point_size=15, render_points_as_spheres=True)
    # #     # p.add_mesh(points['input'], color='r', point_size=15, render_points_as_spheres=True)
    # #     # p.add_legend([['icp', 'g'], ['input', 'r']], bcolor=(1.0, 1.0, 1.0))
    # #     # p.add_title('icp estimation', font_size=font_size)
    # #     # p.show_grid()
    # #     cpos = p.show(screenshot='test.png', window_size=(1980, 920))
    # #     print(cpos)
