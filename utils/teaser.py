import pyvista as pv
from pyvista import examples
import numpy as np
from os import makedirs, remove
from os.path import exists, join
from glob import glob
import scipy.io as sio
import argparse
import yaml

def bp():
    import pdb;pdb.set_trace()

import __init__
from global_info import global_info
infos           = global_info()
my_dir          = infos.second_path
project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories
delta_R         = infos.delta_R
delta_T         = infos.delta_T

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

def get_axis(axis_only=False):
    x_axes = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), tip_length=0.25, tip_radius=0.1, tip_resolution=20, shaft_radius=0.05, shaft_resolution=20, scale=0.25)
    y_axes = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(0.0, 1.0, 0.0),  tip_length=0.25, tip_radius=0.1, tip_resolution=20, shaft_radius=0.05, shaft_resolution=20, scale=0.25)
    z_axes = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),  tip_length=0.25, tip_radius=0.1, tip_resolution=20, shaft_radius=0.05, shaft_resolution=20, scale=0.25)
    actor1 = p.add_mesh(x_axes, color='r')
    actor2 = p.add_mesh(y_axes, color='g')
    actor3 = p.add_mesh(z_axes, color='b')

    # add circles
    if not axis_only:
        x_circle = pv.Circle(radius=0.25)
        actor4 = p.add_mesh(x_circle, color='r', show_edges=True, line_width=3, style='wireframe', opacity=0.5)
        y_circle = pv.Circle(radius=0.25)
        y_circle.rotate_y(90)
        actor5 = p.add_mesh(y_circle, color='g', show_edges=True, line_width=3, style='wireframe', opacity=0.5)
        z_circle = pv.Circle(radius=0.25)
        z_circle.rotate_x(90)
        actor6 = p.add_mesh(z_circle, color='b', show_edges=True, line_width=3, style='wireframe', opacity=0.5)
        return [x_axes, y_axes, z_axes, x_circle, y_circle, z_circle], [actor1, actor2, actor3, actor4, actor5, actor6]
    else:
        return [x_axes, y_axes, z_axes], [actor1, actor2, actor3]

def get_cube():
    nx, ny, nz = 3, 3, 3
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    z = np.linspace(0, nz, nz)
    xv, yv, zv = np.meshgrid(x, y, z)
    cubes = []
    actors = []
    for i in range(27):
        mesh = pv.Cube(center=(xv.flatten()[i], yv.flatten()[i], zv.flatten()[i]))
        cubes.append(mesh)
        actor = p.add_mesh(mesh, color='r', cmap='plasma', metallic=1.0, roughness=0.6) # color='r',
        actors.append(actor)
    return cubes, actors

def get_mesh_rgb(point_cloud):
    # use the same mesh. but add colors according to canonical coordinates
    coords = np.copy(point_cloud.points)
    normalized_xyz = np.absolute(coords)[:, 0:1] / np.max(np.absolute(coords)[:, 0:1])
    gt_colors = (normalized_xyz * np.array([[255,223,0]]) ).astype(np.uint8)
    actor = p.add_mesh(point_cloud, scalars=gt_colors, rgb=True,  cmap='plasma', point_size=10, render_points_as_spheres=True) #
    return actor

def get_mesh_vector(point_cloud):
    # copy and get a new one
    mesh_equi = point_cloud.copy()
    actor1 = p.add_mesh(mesh_equi, color='r', point_size=4, render_points_as_spheres=True)

    mesh_equi['scalars'] = 50 * np.random.rand(mesh_equi.n_points)
    mesh_equi['orient'] = mesh_equi.points / np.linalg.norm(mesh_equi.points, axis=-1, keepdims=True)
    total_num = mesh_equi.points.shape[0]
    mask = np.random.permutation(total_num)[:int(total_num/8*7)]
    mesh_equi['scalars'][mask] = 0
    geom = pv.Arrow()  # This could be any dataset
    glyphs = mesh_equi.glyph(orient="orient", scale="scalars", factor=0.003, geom=geom) #
    actor2 = p.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='plasma')

    return [mesh_equi, glyphs], [actor1, actor2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_entry', default='complete_airplane', help='yyds')
    args = parser.parse_args()
    np.random.seed(0)
    pv.set_plot_theme("document")
    target_entry    = args.target_entry
    cfg = simple_config(target_entry)

    dx, dy = 10, 0
    instance_ids = ['0627', '0670', '0705', '0715', '0726']
    pclouds = []
    for instance_id in instance_ids:
        complete_fname = cfg.canonical_path + f'/{cfg.category}/testR/{cfg.category}_{instance_id}.mat'
        pc = sio.loadmat(complete_fname)['pc']
        boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        pc_canon = (pc - center_pt.reshape(1, 3))/length_bb
        pc_canon = pc_canon # NOCS space
        pc_canon = np.random.permutation(pc_canon)[:1936] # 1936
        pclouds.append(pc_canon)
    print([x.shape for x in pclouds])

    p = pv.Plotter(notebook=0, shape=(3, 4), border=False)
    p.open_gif(f'./imgs/{instance_id}.gif')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> stage 1: problem settings <<<<<<<<<<<<<<<<<<<<<<<<<<<< #
    """
    get performers
    """
    pc_canon = pclouds[0]
    p.subplot(1,0)
    p.add_text('Input point clouds', position=(200, 500), color='blue', shadow=True, font_size=26)

    point_cloud = pv.PolyData(pc_canon.astype(np.float32)[:, :3])
    mesh_inv = point_cloud.copy()
    actor_mesh = p.add_mesh(point_cloud, color='r', point_size=10, render_points_as_spheres=True) #F

    p.subplot(1,3)
    actor_t = p.add_text('6D Pose estimation', position=(200, 500), color='blue', shadow=True, font_size=26)
    actor0 = p.add_mesh(point_cloud, color='r', point_size=10, render_points_as_spheres=True)
    helpers_axis, actors_axis = get_axis()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()

    performers = [point_cloud]
    helpers= helpers_axis
    r_arr = np.array([0, 30, 60, 60, 90, 120])
    t_arr = np.zeros((7, 3))

    for j in range(3):
        p.subplot(0,3)
        if j == 0:
            actor = p.add_text('no GT annotations', position=(100, 400), color='r', shadow=True, font_size=26)

        if j == 1:
            actor = p.add_text('no CAD models', position=(100, 300), color='r', shadow=True, font_size=26)

        if j == 2:
            actor = p.add_text('no multi-view supervisions', position=(100, 200), color='r', shadow=True, font_size=26)
        p.show(auto_close=False, full_screen=True)
        p.write_frame()

    for j in range(1):
        p.subplot(0,1)
        actor_text_ss = p.add_text('Self-supervised', position=(200, 300), color='k', shadow=True, font_size=50)

        p.subplot(0,2)
        actor_text_l = p.add_text('Learning', position=(150, 300), color='k', shadow=True, font_size=50)

        p.show(auto_close=False, full_screen=True)
        p.write_frame()

    for j in range(r_arr.shape[0]):
        for performer in performers:
            performer.rotate_y(r_arr[j])
        for helper in helpers:
            helper.rotate_y(r_arr[j])
        p.show(auto_close=False, full_screen=True)
        p.write_frame()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> stage 2: key methods <<<<<<<<<<<<<<<<<<<<<<<<<<<< #
    """
    get performers EQN, invariant feature, equivariant feature, feature activation, pose estimation
    """
    for actor in [actor_t, actor0] + actors_axis:
        p.remove_actor(actor)
    p.subplot(1,1)
    p.add_text('___', position=(0, 300), color='blue', shadow=True, font_size=50)
    p.add_text('>', position=(110, 270), color='blue', shadow=True, font_size=40)
    actor_text_eqn = p.add_text('Equivariant Neural Network', position=(100, 500), color='blue', shadow=True, font_size=26)
    cubes, cubes_actors = get_cube()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()

    p.subplot(2,2) # same point cloud, rotate with
    p.add_text('___', position=(0, 300), color='blue', shadow=True, font_size=50)
    p.add_text('>', position=(110, 270), color='blue', shadow=True, font_size=40)
    actor_text_inv = p.add_text('Invariant feature', position=(200, 500), color='blue', shadow=True, font_size=26)
    actor_mesh_inv = get_mesh_rgb(point_cloud)
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()

    p.subplot(2,3) # copy() and no need to move, only to delete
    p.add_text('___', position=(0, 300), color='blue', shadow=True, font_size=50)
    p.add_text('>', position=(110, 270), color='blue', shadow=True, font_size=40)
    actor_text_shape = p.add_text('Canonical Shape', position=(200, 500), color='blue', shadow=True, font_size=26)
    actor_mesh_shape = p.add_mesh(mesh_inv, color='r', point_size=6, render_points_as_spheres=True)
    _, _ = get_axis(axis_only=True) # unchanged

    # plot it with coordinate axis
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()

    p.subplot(1,2) # copy() and new control
    p.add_text('___', position=(0, 300), color='blue', shadow=True, font_size=50)
    p.add_text('>', position=(110, 270), color='blue', shadow=True, font_size=40)
    actor_text_equi = p.add_text('Equivariant feature', position=(200, 500), color='blue', shadow=True, font_size=26)
    mesh_equis, actor_mesh_equis = get_mesh_vector(point_cloud)
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()

    p.subplot(1,3)
    p.add_text('___', position=(0, 300), color='blue', shadow=True, font_size=50)
    p.add_text('>', position=(110, 270), color='blue', shadow=True, font_size=40)
    actor_text_pose = p.add_text('6D Pose Estimation', position=(200, 500), color='blue', shadow=True, font_size=26)
    actor0 = p.add_mesh(point_cloud, color='r', point_size=10, render_points_as_spheres=True)
    helpers_axis, actors_axis = get_axis()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()

    p.subplot(2,0)
    actor_text_loss = p.add_text('Reconstruction', position=(380, 300), color='r', shadow=True, font_size=40)

    p.subplot(2,1)
    actor_text_l = p.add_text('Loss', position=(50, 300), color='r', shadow=True, font_size=40)
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    p.show(auto_close=False, full_screen=True)
    p.write_frame()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> stage 3: get different poses, get different shapes and poses <<<<<<<<<<<<<<<<<<<<<<<<<<<< #
    """
    remove existing point cloud and vectors,
    get new performers
    """
    p.subplot(0,0)
    actor_text_robust = p.add_text("Handling ", position=(200, 250), color='k', shadow=True, font_size=30)
    texts = ['random poses', 'shape variations', 'shape variations', 'shape variations', 'shape variations', 'partialness', 'partialness', 'partialness', 'partialness', 'partialness', 'partialness',  'partialness', 'partialness']
    for i in range(11):
        # clear existing helpers
        if i >0:
            to_clears = [actor_text_indicator, actor_mesh_inv, actor_mesh_shape] + actor_mesh_equis + actors_axis
            for j, actor in enumerate(to_clears):
                print(f'Now: {j}')
                p.remove_actor(actor)
            if i > 4:
                if i == 5:
                    raw_pts = pclouds[4]
                    subsampled_pts = raw_pts[raw_pts[:, 1]>-0.1]
                elif i == 6:
                    raw_pts = pclouds[4]
                    subsampled_pts = raw_pts[raw_pts[:, 1]>-0.1]
                    subsampled_pts = subsampled_pts[subsampled_pts[:, 0]>-0.3]
                elif i == 7:
                    raw_pts = pclouds[4]
                    subsampled_pts = raw_pts[raw_pts[:, 1]>-0.1]
                    subsampled_pts = subsampled_pts[subsampled_pts[:, 0]>-0.15]
                elif i == 8:
                    raw_pts = pclouds[3]
                    subsampled_pts = raw_pts[raw_pts[:, 1]<0.1]
                elif i == 9:
                    raw_pts = pclouds[3]
                    subsampled_pts = raw_pts[raw_pts[:, 1]<0.1]
                    subsampled_pts = subsampled_pts[subsampled_pts[:, 0]>-0.3]
                elif i == 10:
                    raw_pts = pclouds[3]
                    subsampled_pts = raw_pts[raw_pts[:, 1]<0.1]
                    subsampled_pts = subsampled_pts[subsampled_pts[:, 0]>-0.15]
                n_pts = subsampled_pts.shape[0]
                n_tiles = int(1936 / n_pts)
                n_left = 1936 - n_tiles * n_pts
                subsampled_pts = np.concatenate([subsampled_pts] * n_tiles + [subsampled_pts[:n_left]], axis=0)
                point_cloud.points = subsampled_pts
                mesh_equis[0].points = subsampled_pts
                mesh_inv.points = raw_pts
            else:
                point_cloud.points = pclouds[i]
                mesh_equis[0].points = pclouds[i]
                mesh_inv.points = pclouds[i]

            # create new performers
            p.subplot(2,2)
            actor_mesh_inv = get_mesh_rgb(point_cloud)

            p.subplot(2,3) # remain unchanged
            actor_mesh_shape = p.add_mesh(mesh_inv, color='r', point_size=6, render_points_as_spheres=True)

            p.subplot(1,2)
            mesh_equis, actor_mesh_equis = get_mesh_vector(point_cloud)

            p.subplot(1,3)
            actor0 = p.add_mesh(point_cloud, color='r', point_size=10, render_points_as_spheres=True)
            helpers_axis, actors_axis = get_axis()

        p.subplot(0,0)
        actor_text_indicator = p.add_text(texts[i], position=(200, 100), color='r', shadow=True, font_size=40)

        performers = [point_cloud] + mesh_equis
        helpers= helpers_axis

        r_arr = np.arange(0, 180, 30)
        t_arr = np.zeros((5, 3))
        iter_num = 5
        if i > 0:
            iter_num = 2
            r_arr = np.ones((10))
        for j in range(iter_num):
            for performer in performers:
                performer.rotate_y(r_arr[j])
            for helper in helpers:
                helper.rotate_y(r_arr[j])
            p.show(auto_close=False, full_screen=True)
            p.write_frame()
            p.show(auto_close=False, full_screen=True)
            p.write_frame()
    p.close()
