import matplotlib
# matplotlib.use('Agg')
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import PatchCollection

import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from descartes import PolygonPatch

import matplotlib.pyplot as plt  # matplotlib.use('Agg') # TkAgg
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import pyrender
import trimesh
from scipy.spatial import Delaunay

import os
import cv2

""""
render things all together in one window with multiple independent subplots
3d: plot3d_pts(pts, pts_name, s=1,
joints: visualize_joints_2d, plot_skeleton
arrows: plot_arrows(points, offset=None, joint=None)
hands: plot_hand_w_object
mesh:  visualize_mesh
imgs:  plot_imgs


"""
def bp():
    import pdb;pdb.set_trace()

# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(30, 45)

    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img


def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels, out_file=None, show=False):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_2d(img,
                 hand_joints=None,
                 hand_verts=None,
                 obj_verts=None,
                 links=[(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                        (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    if hand_joints is not None:
        visualize_joints_2d(
            ax, hand_joints, joint_idxs=False, links=links)
    if obj_verts is not None:
        ax.scatter(obj_verts[:, 0], obj_verts[:, 1], alpha=0.1, c='r')
    if hand_verts is not None:
        ax.scatter(hand_verts[:, 0], hand_verts[:, 1], alpha=0.1, c='b')
    plt.show()

def visualize_3d(img,
                 hand_verts=None,
                 hand_faces=None,
                 obj_verts=None,
                 obj_faces=None):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(img)
    ax.axis('off')
    ax = fig.add_subplot(122, projection='3d')
    add_mesh(ax, hand_verts, hand_faces)
    add_mesh(ax, obj_verts, obj_faces, c='r')
    cam_equal_aspect_3d(ax, hand_verts)
    plt.show()

def visualize_pointcloud(points, normals=None, labels=None,
                         title_name='0', out_file=None, backend='matplotlib', show=False):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    if backend == 'pyrender':
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        oc = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
        scene = pyrender.Scene()
        if type(points) is not list:
            points = [points]
        if type(labels) is not list:
            labels = [labels] + [None] * (len(points) - 1)
        palette = get_tableau_palette()
        for j, pts in enumerate(points):
            if pts.shape[0] > 5:
                if labels[j] is not None and labels[j].shape[0] == pts.shape[0]:
                    colors = np.array(labels[j]).reshape(-1, 1) * palette[j:j+1] / 255.0
                else:
                    colors = np.ones(pts.shape)
                    colors = colors * palette[j:j+1] / 255.0
                cloud = pyrender.Mesh.from_points(pts, colors=colors)
                scene.add(cloud)
            else:
                # boundary_pts = [np.min(np.array(pts), axis=0), np.max(np.array(pts), axis=0)]
                # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
                # print(length_bb/100)
                sm = trimesh.creation.uv_sphere(radius=1)
                sm.visual.vertex_colors = (palette[j]/255.0).tolist()
                # sm.visual.vertex_colors = [1.0, 1.0, 0.0]
                tfs = np.tile(np.eye(4), (len(pts), 1, 1))
                tfs[:,:3,3] = pts
                m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(m)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size = (1920, 1024), point_size=15, show_world_axis=True, window_title=title_name)
        return scene, viewer

    elif backend == 'matplotlib':
        points = np.asarray(points)
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)

        if labels is not None:
            points = points[np.where(labels>0)[0]]
        ax.scatter(points[:, 2], points[:, 0], points[:, 1])

        # normals
        if normals is not None:
            ax.quiver(
                points[:, 2], points[:, 0], points[:, 1],
                normals[:, 2], normals[:, 0], normals[:, 1],
                length=0.1, color='k'
            )
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.view_init(elev=30, azim=45)
        if title_name is not None:
            plt.title(title_name)
        if out_file is not None:
            plt.savefig(out_file)
        if show:
            plt.show()
        plt.close(fig)

def visualize_mesh(input, pts=None, labels=None, ax=None, mode='mesh', backend='matplotlib', title_name=None, viz_mesh=True):
    """
    we have backend of matplotlib, pyrender
    """
    if mode == 'file':
        file_name = input
        with open(file_name , 'r') as obj_f:
            mesh_dict = fast_load_obj(obj_f)[0]
        mesh = trimesh.load(mesh_dict)
    elif mode == 'trimesh':
        if isinstance(input, list):
            mesh = input[0]
        else:
            mesh = input
        mesh_dict = {}
        mesh_dict['vertices'] = mesh.vertices
        mesh_dict['faces'] = mesh.faces
    else:
        mesh_dict = input
        mesh = trimesh.load(mesh_dict)
    v, f = mesh.vertices, mesh.faces
    # get mesh_dict
    if backend=='matplotlib':
        tri = Delaunay(mesh_dict['vertices'])
        dmesh = Poly3DCollection(
            mesh_dict['vertices'][tri.simplices[:, :3]], alpha=0.5)
        dmesh.set_edgecolor('b')
        dmesh.set_facecolor('r')
        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(121, projection='3d')
        ax.add_collection3d(dmesh)
    elif backend=='pyrender':
        scene = pyrender.Scene()
        if pts is not None:
            if pts.shape[0] > 5000:
                if labels is not None:
                    colors = np.zeros(pts.shape)
                    colors[labels < 0.1, 2] = 1
                    colors[labels > 0.1, 1] = 1
                    cloud = pyrender.Mesh.from_points(pts, colors=colors)
                else:
                    cloud = pyrender.Mesh.from_points(pts)
                scene.add(cloud)
            else:
                sm = trimesh.creation.uv_sphere(radius=0.01)
                sm.visual.vertex_colors = [1.0, 1.0, 0.0]
                tfs = np.tile(np.eye(4), (len(pts), 1, 1))
                tfs[:,:3,3] = pts
                m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(m)
        if viz_mesh:
            print('viz mesh', mesh)
            mesh_vis = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh_vis)
        else:
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (len(v), 1, 1))
            tfs[:,:3,3] = v
            m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(m)

        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5, show_world_axis=True, window_title=title_name)
        # record=True, run_in_thread=True,
        # t0 = time.time()
        # plt.pause(2)
        # viewer.close_external()
        # viewer.save_gif('./demo.gif')
    # else: # meshplot
    #     # meshplot.offline()
    #     # p = plot(v, f, shading={"point_size": 0.2})
    #     # p.add_points(pts,shading={"point_size": 0.02, "point_color": "blue"})
    #     # p.save("test2.html")
    #     # p.add_edges(v_box, f_box, shading={"line_color": "red"});
    #     # add_edges(vertices, edges, shading={}, obj=None)
    #     # add_lines(beginning, ending, shading={}, obj=None)
    #     # add_mesh(v, f, c=None, uv=None, shading={})
    #     # add_points(points, shading={}, obj=None)
    #     # add_text(text, shading={})
    #     # remove_object(obj_id)
    #     # reset()
    #     # to_html()
    #     # update()


def visualize_joints_2d(
    ax, joints, joint_idxs=True, links=None, alpha=1, scatter=True, linewidth=2
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha, linewidth=linewidth)
    ax.axis("equal")


def _draw2djoints(ax, annots, links, alpha=1, linewidth=1):
    colors = ["r", "m", "b", "c", "g"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth,
            )


def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linewidth=linewidth,
    )


def visualize_joints_2d_cv2(
    img,
    joints,
    joint_idxs=True,
    links=None,
    alpha=1,
    scatter=True,
    linewidth=2,
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image

    # Add idx labels to joints
    _draw2djoints_cv2(img, joints, links, alpha=alpha, linewidth=linewidth)
    return img


def _draw2djoints_cv2(img, annots, links, alpha=1, linewidth=1):
    colors = [
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255),
        (255, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            img = _draw2dseg_cv2(
                img,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                col=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth,
            )
    return img


def _draw2dseg_cv2(
    img, annot, idx1, idx2, col=(0, 255, 0), alpha=1, linewidth=1
):
    cv2.line(
        img,
        (annot[idx1, 0], annot[idx1, 1]),
        (annot[idx2, 0], annot[idx2, 1]),
        col,
        linewidth,
    )
    return img

def get_hand_line_ids():
    line_ids = []
    for finger in range(5):
        base = 4*finger + 1
        line_ids.append([0, base])
        for j in range(3):
            line_ids.append([base+j, base+j+1])
    line_ids = np.asarray(line_ids, dtype=int)

    return line_ids


def plot_skeleton(hand_joints, line_ids):
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    top     = plt.cm.get_cmap('Oranges_r', 128)
    bottom  = plt.cm.get_cmap('Blues', 128)

    colors = np.vstack((top(np.linspace(0, 1, 10)),
                           bottom(np.linspace(0, 1, 10))))
    # colors = ListedColormap(newcolors, name='OrangeBlue')
    # colors  = cmap(np.linspace(0., 1., 5))
    # colors = ['Blues', 'Blues',  'Blues', 'Blues', 'Blues']
    all_poss=['o', 'o','o']
    num     = len(pts)
    ax = plt.subplot(1, 1, 1, projection='3d')
    if view_angle==None:
        ax.view_init(elev=36, azim=-49)
    else:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    n=0
    ax.scatter(hand_joints[:, 0],  hand_joints[:, 1], hand_joints[:, 2], marker=all_poss[n], s=ss[n], cmap=colors[n], label='hand joints')

    line_ids = get_hand_line_ids()

    for line_idx, (idx0, idx1) in enumerate(line_ids):
        bone = hand_joints[idx0] - hand_joints[idx1]
        h = np.linalg.norm(bone)

def plot3d_pts(pts, pts_name, s=1, dpi=350, title_name=None, sub_name='default', arrows=None, \
                    color_channel=None, colorbar=False, limits=None,\
                    bcm=None, puttext=None, view_angle=None,\
                    save_fig=False, save_path=None, flip=True,\
                    axis_off=False, show_fig=True, mode='pending'):
    """
    fig using,
    """
    fig     = plt.figure(dpi=dpi)
    # cmap    = plt.cm.jet
    top     = plt.cm.get_cmap('Oranges_r', 128)
    bottom  = plt.cm.get_cmap('Blues', 128)
    if isinstance(s, list):
        ss = s
    else:
        ss = [s] * len(pts[0])
    colors = np.vstack((top(np.linspace(0, 1, 10)),
                           bottom(np.linspace(0, 1, 10))))
    # colors = ListedColormap(newcolors, name='OrangeBlue')
    # colors  = cmap(np.linspace(0., 1., 5))
    # '.', '.', '.',
    all_poss=['o', 'o', 'o', 'o','o', '*', '.','o', 'v','^','>','<','s','p','*','h','H','D','d','1','','']
    c_set   = ['r', 'b', 'g', 'k', 'm']
    num     = len(pts)
    for m in range(num):
        ax = plt.subplot(1, num, m+1, projection='3d')
        if view_angle==None:
            ax.view_init(elev=36, azim=-49)
        else:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
        # if len(pts[m]) > 1:
        for n in range(len(pts[m])):
            if color_channel is None:
                ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[n], s=ss[n], cmap=colors[n], label=pts_name[m][n], depthshade=False)
            else:
                if len(color_channel[m][n].shape) < 2:
                    color_channel[m][n] = color_channel[m][n][:, np.newaxis] * np.array([[1]])
                if np.amax(color_channel[m][n], axis=0, keepdims=True)[0, 0] == np.amin(color_channel[m][n], axis=0, keepdims=True)[0, 0]:
                    rgb_encoded = color_channel[m][n]
                else:
                    rgb_encoded = (color_channel[m][n] - np.amin(color_channel[m][n], axis=0, keepdims=True))/np.array(np.amax(color_channel[m][n], axis=0, keepdims=True) - np.amin(color_channel[m][n], axis=0, keepdims=True)+ 1e-6)
                if len(pts[m])==3 and n==2:
                    p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[4], s=ss[n], c=rgb_encoded, label=pts_name[m][n], depthshade=False)
                else:
                    p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[n], s=ss[n], c=rgb_encoded, label=pts_name[m][n], depthshade=False)
                if colorbar:
                    fig.colorbar(p)
            if arrows is not None:
                points, offset_sub = arrows[m][n]['p'], arrows[m][n]['v']
                offset_sub = offset_sub
                if len(points.shape) < 2:
                    points = points.reshape(-1, 3)
                if len(offset_sub.shape) < 2:
                    offset_sub = offset_sub.reshape(-1, 3)
                ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset_sub[:, 0], offset_sub[:, 1], offset_sub[:, 2], color=c_set[n], linewidth=0.2)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        if axis_off:
            plt.axis('off')

        if title_name is not None:
            if len(pts_name[m])==1:
                plt.title(title_name[m]+ ' ' + pts_name[m][0] + '    ')
            else:
                plt.legend(loc=0)
                plt.title(title_name[m]+ '    ')

        if bcm is not None:
            for j in range(len(bcm)):
                ax.plot3D([bcm[j][0][0], bcm[j][2][0], bcm[j][6][0], bcm[j][4][0], bcm[j][0][0]], \
                    [bcm[j][0][1], bcm[j][2][1], bcm[j][6][1], bcm[j][4][1], bcm[j][0][1]], \
                    [bcm[j][0][2], bcm[j][2][2], bcm[j][6][2], bcm[j][4][2], bcm[j][0][2]], 'blue')

                ax.plot3D([bcm[j][1][0], bcm[j][3][0], bcm[j][7][0], bcm[j][5][0], bcm[j][1][0]], \
                    [bcm[j][1][1], bcm[j][3][1], bcm[j][7][1], bcm[j][5][1], bcm[j][1][1]], \
                    [bcm[j][1][2], bcm[j][3][2], bcm[j][7][2], bcm[j][5][2], bcm[j][1][2]], 'gray')

                for pair in [[0, 1], [2, 3], [4, 5], [6, 7]]:
                    ax.plot3D([bcm[j][pair[0]][0], bcm[j][pair[1]][0]], \
                        [bcm[j][pair[0]][1], bcm[j][pair[1]][1]], \
                        [bcm[j][pair[0]][2], bcm[j][pair[1]][2]], 'red')
        if puttext is not None:
            ax.text2D(0.55, 0.80, puttext, transform=ax.transAxes, color='blue', fontsize=6)
        # if limits is None:
        #     limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
        # set_axes_equal(ax, limits=limits)
        # cam_equal_aspect_3d(ax, np.concatenate(pts[0], axis=0), flip_x=flip, flip_y=flip)
    if show_fig:
        if mode == 'continuous':
            plt.draw()
        else:
            plt.show()

    if save_fig:
        if save_path is None:
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(sub_name, title_name[0]), pad_inches=0)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name[0]), pad_inches=0)
    if mode != 'continuous':
        plt.close()


def plot_hand_w_object(obj_verts=None, obj_faces=None, hand_verts=None, hand_faces=None, s=5, pts=None, jts=None, nmls=None, viz_m=False, viz_c=False, viz_j=False, viz_n=False, save_path=None, flip=True, save=False, mode='keyboard'):
    """
    Functions taken from the ObMan dataset repo (https://github.com/hassony2/obman)
    """
    colors = ['r']*len(hand_faces) + ['b']*len(obj_faces)

    frames = []
    fig = plt.figure()

    cmap    = plt.cm.jet
    top     = plt.cm.get_cmap('Oranges_r', 128)
    bottom  = plt.cm.get_cmap('Blues', 128)
    colors_full = np.vstack((top(np.linspace(0, 1, 10)),
                           bottom(np.linspace(0, 1, 10))))
    all_poss=['o', 'o','o', '.','o', '*', '.','o', 'v','^','>','<','s','p','*','h','H','D','d','1','','']


    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(111, projection='3d')

    verts = hand_verts
    add_group_meshs(ax, np.concatenate((verts, obj_verts)), np.concatenate((hand_faces, obj_faces + verts.shape[0])), alpha=1, c=colors)
    if pts is not None:
        for m in range(len(pts)):
            ss = [s] * len(pts[m])
            ss[0] = 10**2
            for n in range(len(pts[m])):
                ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[n], s=ss[n]**2, cmap=colors[n])
    if pts is not None:
        cam_equal_aspect_3d(ax, np.concatenate((verts, pts[0][0])), flip_x=flip, flip_y=flip)
    else:
        cam_equal_aspect_3d(ax, verts, flip_x=flip, flip_y=flip)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if mode=='keyboard':
        pressed_keyboard = False
        while not pressed_keyboard:
            pressed_keyboard = plt.waitforbuttonpress()
    elif mode =='continuous':
        plt.draw()
    else:
        plt.show(block=False)
        plt.pause(1)

    if save:
        fig.savefig(save_path, pad_inches=0)
    if mode !='continuous':
        plt.close(fig)

    return

def add_mesh(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == 'b':
        face_color = (141 / 255, 184 / 255, 226 / 255)
    elif c == 'r':
        face_color = (226 / 255, 184 / 255, 141 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

def add_group_meshs(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = []
    for i in range(len(c)):
        if c[i] == 'b':
            face_color.append((141 / 255, 184 / 255, 226 / 255))
        elif c[i] == 'r':
            face_color.append((226 / 255, 184 / 255, 141 / 255))
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)


def viz_voxels():
    # use openGL to visualize voxels
    pass

def plot_scene_w_grasps(list_obj_verts, list_obj_faces, list_obj_handverts, list_obj_handfaces, plane_parameters):
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins

    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    # We will convert this into a single mesh, and then use add_group_meshs to plot it in 3D
    allverts = np.zeros((0,3))
    allfaces = np.zeros((0,3))
    colors = []
    for i in range(len(list_obj_verts)):
        allfaces = np.concatenate((allfaces, list_obj_faces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_verts[i]))
        colors = np.concatenate((colors, ['r']*len(list_obj_faces[i])))

    for i in range(len(list_obj_handverts)):
        allfaces = np.concatenate((allfaces, list_obj_handfaces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_handverts[i]))
        colors = np.concatenate((colors, ['b']*len(list_obj_handfaces[i])))

    allfaces = np.int32(allfaces)
    print(np.max(allfaces))
    print(np.shape(allverts))
    add_group_meshs(ax, allverts, allfaces, alpha=1, c=colors)

    cam_equal_aspect_3d(ax, np.concatenate(list_obj_verts, 0), flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    # Show plane too:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    step = 0.05
    border = 0.0 #step
    X, Y = np.meshgrid(np.arange(xlim[0]-border, xlim[1]+border, step),
               np.arange(ylim[0]-border, ylim[1]+border, step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
      for c in range(X.shape[1]):
        Z[r, c] = (-plane_parameters[0] * X[r, c] - plane_parameters[1] * Y[r, c] + plane_parameters[3])/plane_parameters[2]
    ax.plot_wireframe(X, Y, Z, color='r')

    pressed_keyboard = False
    while not pressed_keyboard:
        pressed_keyboard = plt.waitforbuttonpress()

    plt.close(fig)

def cam_equal_aspect_3d(ax, verts, flip_x=False, flip_y=False, flip_z=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    if flip_y:
        ax.set_ylim(centers[1] + r, centers[1] - r)
    else:
        ax.set_ylim(centers[1] - r, centers[1] + r)

    if flip_z:
        ax.set_zlim(centers[2] + r, centers[2] - r)
    else:
        ax.set_zlim(centers[2] - r, centers[2] + r)



def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)

def get_tableau_palette():
    palette = np.array([[ 78,121,167], # blue
                        [255, 87, 89], # red
                        [ 89,169, 79], # green
                        [242,142, 43], # orange
                        [237,201, 72], # yellow
                        [176,122,161], # purple
                        [255,157,167], # pink
                        [118,183,178], # cyan
                        [156,117, 95], # brown
                        [186,176,172]  # gray
                        ],dtype=np.uint8)
    return palette

def set_axes_equal(ax, limits=None):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    if limits is None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    else:
        x_limits, y_limits, z_limits = limits
        ax.set_xlim3d([x_limits[0], x_limits[1]])
        ax.set_ylim3d([y_limits[0], y_limits[1]])
        ax.set_zlim3d([z_limits[0], z_limits[1]])

def plot2d_img(imgs, title_name=None, dpi=200, cmap=None, save_fig=False, show_fig=False, save_path=None, sub_name='0'):
    all_poss=['.','o','v','^']
    num     = len(imgs)
    for m in range(num):
        ax = plt.subplot(1, num, m+1)
        if cmap is None:
            plt.imshow(imgs[m])
        else:
            plt.imshow(imgs[m], cmap=cmap)
        plt.title(title_name[m])
        plt.axis('off')
        plt.grid('off')
    if show_fig:
        plt.show()
    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig('{}/{}.png'.format(save_path, sub_name, pad_inches=0))

def plot_imgs(imgs, imgs_name, title_name='default', sub_name='default', save_path=None, save_fig=False, axis_off=False, show_fig=True, dpi=150):
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    num = len(imgs)
    for m in range(num):
        ax1 = plt.subplot(1, num, m+1)
        plt.imshow(imgs[m].astype(np.uint8))
        if title_name is not None:
            plt.title(title_name[0]+ ' ' + imgs_name[m])
        else:
            plt.title(imgs_name[m])
    if show_fig:
        plt.show()
    if save_fig:
        if save_path is None:
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(sub_name, title_name[0]), pad_inches=0)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name[0]), pad_inches=0)

    plt.close()

def plot_arrows(points, offset=None, joint=None, whole_pts=None, title_name='default', limits=None, idx=None, dpi=200, s=0.1, thres_r=0.1, show_fig=True, sparse=True, index=0, save=False, save_path=None):
    """
    points: [N, 3]
    offset: [N, 3] or list of [N, 3]
    joint : [P0, ll], a list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set   = ['r', 'b', 'g', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    num     = len(points)
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=60, azim=150)
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[0], s=s)
    if whole_pts is not None:
        p = ax.scatter(whole_pts[:, 0], whole_pts[:, 1], whole_pts[:, 2],  marker=all_poss[1], s=s)
    if not isinstance(offset, list):
        offset = [offset]

    for j, offset_sub in enumerate(offset):
        if sparse:
            if idx is None:
                ax.quiver(points[::10, 0], points[::10, 1], points[::10, 2], offset_sub[::10, 0], offset_sub[::10, 1], offset_sub[::10, 2], color=c_set[j])
            else:
                points = points[idx, :]
                ax.quiver(points[::2, 0], points[::2, 1], points[::2, 2], offset_sub[::2, 0], offset_sub[::2, 1], offset_sub[::2, 2], color=c_set[j])
        else:
            if idx is None:
                ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset_sub[:, 0], offset_sub[:, 1], offset_sub[:, 2], color=c_set[j])
            else:
                print(idx)
                ax.quiver(points[idx[:], 0], points[idx[:], 1], points[idx[:], 2], offset_sub[idx[:], 0], offset_sub[idx[:], 1], offset_sub[idx[:], 2], color=c_set[j])
    if joint is not None:
        for sub_j in joint:
            length = 0.5
            sub_j[0] = sub_j[0].reshape(1,3)
            sub_j[1] = sub_j[1].reshape(-1)
            ax.plot3D([sub_j[0][0, 0]- length * sub_j[1][0], sub_j[0][0, 0] + length * sub_j[1][0]], \
                      [sub_j[0][0, 1]- length * sub_j[1][1], sub_j[0][0, 1] + length * sub_j[1][1]], \
                      [sub_j[0][0, 2]- length * sub_j[1][2], sub_j[0][0, 2] + length * sub_j[1][2]],  'blue', linewidth=8)
    if limits is None:
        limits = [[-1, 1], [-1, 1], [-1, 1]]
    set_axes_equal(ax, limits=limits)
    plt.title(title_name)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    if show_fig:
        plt.show()

    if save:
        if save_path is None:
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(index, title_name[0]), pad_inches=0)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, index, title_name[0]), pad_inches=0)
    plt.close()

def plot_lines(orient_vect):
    """
    orient_vect: list of [3] or None
    """
    fig     = plt.figure(dpi=150)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set   = ['r', 'b', 'g', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=32, azim=-54)
    for sub_j in orient_vect:
        if sub_j is not None:
            length = 0.5
            ax.plot3D([0, sub_j[0]], \
                      [0, sub_j[1]], \
                      [0, sub_j[2]],  'blue', linewidth=5)
    plt.show()
    plt.close()

def plot_arrows_list(points_list, offset_list, whole_pts=None, title_name='default', limits=None, dpi=200, s=5, lw=1, length=0.5, view_angle=None, sparse=True, axis_off=False):
    """
    points: list of [N, 3]
    offset: nested list of [N, 3]
    joint : [P0, ll], 2-order nested list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set = ['r', 'g', 'b', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    if view_angle==None:
        ax.view_init(elev=36, azim=-49)
    else:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    if whole_pts is not None:
        p = ax.scatter(whole_pts[:, 0], whole_pts[:, 1], whole_pts[:, 2],  marker=all_poss[0], s=s)
    for i in range(len(points_list)):
        points = points_list[i]
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[1], s=10,  cmap=colors[i+1])
        offset = offset_list[i]
        if sparse:
            ls =5
            ax.quiver(points[::ls, 0], points[::ls, 1], points[::ls, 2], offset[::ls, 0], offset[::ls, 1], offset[::ls, 2], color=c_set[i], linewidth=lw)
        else:
            ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset[:, 0], offset[:, 1], offset[:, 2], color='r', linewidth=lw)
    if limits is None:
        limits = []
    set_axes_equal(ax, limits=limits)
    plt.title(title_name)
    if axis_off:
        plt.axis('off')
        plt.grid('off')
    plt.show()
    plt.close()

def plot_joints_bb_list(points_list, offset_list=None, joint_list=None, whole_pts=None, bcm=None, view_angle=None, title_name='default', sub_name='0', dpi=200, s=15, lw=1, length=0.5, sparse=True, save_path=None, show_fig=True, save_fig=False):
    """
    points: list of [N, 3]
    offset: nested list of [N, 3]
    joint : [P0, ll], 2-order nested list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    top     = plt.cm.get_cmap('Oranges_r', 128)
    bottom  = plt.cm.get_cmap('Blues', 128)

    colors = np.vstack((top(np.linspace(0, 1, 10)),
                           bottom(np.linspace(0, 1, 10))))
    c_set = ['g', 'b', 'm', 'y', 'r', 'c']
    all_poss=['.','o','.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    if view_angle==None:
        ax.view_init(elev=36, azim=-49)
    else:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    # ax.view_init(elev=46, azim=-164)
    pts_name = ['part {}'.format(j) for j in range(10)]
    if whole_pts is not None:
        for m, points in enumerate(whole_pts):
            p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[1], s=s, cmap=colors[m], label=pts_name[m])
    center_pt = np.mean(whole_pts[0], axis=0)
    for i in range(len(points_list)):
        points = points_list[i]
        # p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[i], s=s,  c='c')
        if offset_list is not None:
            offset = offset_list[i]# with m previously
            if sparse:
                ax.quiver(points[::50, 0], points[::50, 1], points[::50, 2], offset[::50, 0], offset[::50, 1], offset[::50, 2], color=c_set[i])
            else:
                ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset[:, 0], offset[:, 1], offset[:, 2], color='r')
        # we have two layers
        palette = get_tableau_palette()
        if joint_list is not None:
            if joint_list[i] is not []:
                joint  = joint_list[i] # [[1, 3], [1, 3]]
                jp = joint['p'].reshape(-1)
                jp[1] = 0.5
                jl = joint['l'].reshape(-1)
                # mean
                ax.quiver(jp[0]- 0*length * abs(jl[0]), jp[1]- 0*length * abs(jl[1]), jp[2]- 0*length * abs(jl[2]),\
                        1* length * jl[0], 1 * length * jl[1], 1* length * jl[2], color='black', linewidth=3, arrow_length_ratio=0.3)
                # ax.quiver(jp[0]- 0*length * abs(jl[0]), jp[1]- 0*length * abs(jl[1]), jp[2]- 0*length * abs(jl[2]),\
                #         1* length * jl[0], 1 * length * jl[1], 1* length * jl[2], cmap=palette[0], linewidth=3, arrow_length_ratio=0.3)
    # ax.dist = 8
    print('viewing distance is ', ax.dist)
    if bcm is not None:
        for j in range(len(bcm)):
            color_s = 'gray'
            lw_s =1.5
            # if j == 1:
            #     color_s = 'red'
            #     lw_s = 2
            ax.plot3D([bcm[j][0][0], bcm[j][2][0], bcm[j][6][0], bcm[j][4][0], bcm[j][0][0]], \
                [bcm[j][0][1], bcm[j][2][1], bcm[j][6][1], bcm[j][4][1], bcm[j][0][1]], \
                [bcm[j][0][2], bcm[j][2][2], bcm[j][6][2], bcm[j][4][2], bcm[j][0][2]], color=color_s, linewidth=lw_s)

            ax.plot3D([bcm[j][1][0], bcm[j][3][0], bcm[j][7][0], bcm[j][5][0], bcm[j][1][0]], \
                [bcm[j][1][1], bcm[j][3][1], bcm[j][7][1], bcm[j][5][1], bcm[j][1][1]], \
                [bcm[j][1][2], bcm[j][3][2], bcm[j][7][2], bcm[j][5][2], bcm[j][1][2]], color=color_s, linewidth=lw_s)

            for pair in [[0, 1], [2, 3], [4, 5], [6, 7]]:
                ax.plot3D([bcm[j][pair[0]][0], bcm[j][pair[1]][0]], \
                    [bcm[j][pair[0]][1], bcm[j][pair[1]][1]], \
                    [bcm[j][pair[0]][2], bcm[j][pair[1]][2]], color=color_s, linewidth=lw_s)

    plt.title(title_name, fontsize=10)
    plt.axis('off')
    # plt.legend('off')
    plt.grid('off')
    limits = [[0, 1], [0, 1], [0, 1]]
    set_axes_equal(ax, limits)
    if show_fig:
        plt.show()
    if save_fig:
        if save_path is None:
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(sub_name, title_name), pad_inches=0)
            print('saving figure into ', './results/test/{}_{}.png'.format(sub_name, title_name))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name), pad_inches=0)
            print('saving fig into ', '{}/{}_{}.png'.format(save_path, sub_name, title_name))
    plt.close()

def plot_arrows_list_threshold(points_list, offset_list, joint_list, title_name='default', dpi=200, s=5, lw=5, length=0.5, threshold=0.2):
    """
    points: [N, 3]
    offset: [N, 3]
    joint : [P0, ll], a list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set = ['r', 'g', 'b', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    for i in range(len(points_list)):
        points = points_list[i]
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[n], s=s, c='c')
        if joint_list[i] is not []:
            for m in range(len(joint_list[i])):
                offset = offset_list[i][m]
                joint  = joint_list[i][m]
                offset_norm = np.linalg.norm(offset, axis=1)
                idx = np.where(offset_norm<threshold)[0]
                ax.quiver(points[idx, 0], points[idx, 1], points[idx, 2], offset[idx, 0], offset[idx, 1], offset[idx, 2], color=c_set[i])
                ax.plot3D([joint[0][0, 0]- length * joint[1][0], joint[0][0, 0] + length * joint[1][0]], \
                          [joint[0][0, 1]- length * joint[1][1], joint[0][0, 1] + length * joint[1][1]], \
                          [joint[0][0, 2]- length * joint[1][2], joint[0][0, 2] + length * joint[1][2]],  linewidth=lw, c='blue')
    # set_axes_equal(ax
    plt.title(title_name)
    plt.show()
    plt.close()


def hist_show(values, labels, tick_label, axes_label, title_name, total_width=0.5, dpi=300, save_fig=False, sub_name='seen'):
    """
    labels:
    """
    x     = list(range(len(values[0])))
    n     = len(labels)
    width = total_width / n
    colors=['r', 'b', 'g', 'k', 'y']
    fig = plt.figure(figsize=(20, 5), dpi=dpi)
    ax = plt.subplot(111)

    for i, num_list in enumerate(values):
        if i == int(n/2):
            plt.xticks(x, tick_label, rotation='vertical', fontsize=5)
        plt.bar(x, num_list, width=width, label=labels[i], fc=colors[i])
        if len(x) < 10:
            for j in range(len(x)):
                if num_list[j] < 0.30:
                    ax.text(x[j], num_list[j], '{0:0.04f}'.format(num_list[j]), color='black', fontsize=2)
                else:
                    ax.text(x[j], 0.28, '{0:0.04f}'.format(num_list[j]), color='black', fontsize=2)
        for j in range(len(x)):
            x[j] = x[j] + width
    if title_name.split()[0] == 'rotation':
        ax.set_ylim(0, 30)
    elif title_name.split()[0] == 'translation':
        ax.set_ylim(0, 0.10)
    elif title_name.split()[0] == 'ADD':
        ax.set_ylim(0, 0.10)
    plt.title(title_name)
    plt.xlabel(axes_label[0], fontsize=8, labelpad=0)
    plt.ylabel(axes_label[1], fontsize=8, labelpad=5)
    plt.legend()
    plt.show()
    if save_fig:
        if not os.path.exists('./results/test/'):
            os.makedirs('./results/test/')
        print('--saving fig to ', './results/test/{}_{}.png'.format(title_name, sub_name))
        fig.savefig('./results/test/{}_{}.png'.format(title_name, sub_name), pad_inches=0)
    plt.close()


def draw(img, imgpts, axes=None, color=None):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([1, 3, 7, 5],[3, 7, 5, 1]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 2)


    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip([0, 2, 6, 4],[1, 3, 7, 5]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 2)


    # finally, draw top layer in color
    for i, j in zip([0, 2, 6, 4],[2, 6, 4, 0]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 2)

    # draw axes
    if axes is not None:
        img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last

    return img

def draw_text(draw_image, bbox, text, draw_box=False):
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1

    retval, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)

    bbox_margin = 10
    text_margin = 10

    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2* text_margin) , min(bbox[2] + bbox_margin, 475 - retval[1] - 2* text_margin))
    text_box_pos_br = (text_box_pos_tl[0] + retval[0] + 2* text_margin,  text_box_pos_tl[1] + retval[1] + 2* text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)

    if draw_box:
        cv2.rectangle(draw_image,
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(draw_image,
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255,0,0), -1)

    cv2.rectangle(draw_image,
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0,0,0), 1)

    cv2.putText(draw_image, text, text_pos,
                fontFace, fontScale, (255,255,255), thickness)

    return draw_image

def plot_distribution(d, labelx='Value', labely='Frequency', title_name='Mine', dpi=200, xlimit=None, put_text=False, save_fig=False, sub_name='seen'):
    fig     = plt.figure(dpi=dpi)
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title_name)
    if put_text:
        plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if xlimit is not None:
        plt.xlim(xmin=xlimit[0], xmax=xlimit[1])
    plt.show()
    if save_fig:
        if not os.path.exists('./results/test/'):
            os.makedirs('./results/test/')
        print('--saving fig to ', './results/test/{}_{}.png'.format(title_name, sub_name))
        fig.savefig('./results/test/{}_{}.png'.format(title_name, sub_name), pad_inches=0)
    plt.close()



def viz_err_distri(val_gt, val_pred, title_name):
    if val_gt.shape[1] > 1:
        err = np.linalg.norm(val_gt - val_pred, axis=1)
    else:
        err = np.squeeze(val_gt) - np.squeeze(val_pred)
    plot_distribution(err, labelx='L2 error', labely='Frequency', title_name=title_name, dpi=160)

def draw_line(ax, lines=None):
	if lines is None:
		x = [1,2,3]
		y = [1,2,3]
	else:
		x = lines[0]
		y = lines[1]

	line = Line2D(x, y)
	ax.add_line(line)
	ax.set_xlim(0, max(x))
	ax.set_ylim(0, max(y))
#
# def draw_circle(ax, circles=None, colors=None):
# 	if circles is None:
# 		circles =  [[0.2, 0.2, 0.5]]
# 	for i, params in enumerate(circles):
# 		x, y, r = params
# 		circle = mpatches.Circle([x, y], r, color=colors[i]) #xy1 圆心
# 		ax.add_patch(circle)
# 	plt.axis('equal')

if __name__=='__main__':
    import numpy as np
    d = np.random.laplace(loc=15, scale=3, size=500)
    plot_distribution(d)
