import numpy as np
import transforms3d as tf
from math import pi ,sin, cos, sqrt
import torch
import itertools
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import __init__
from global_info import global_info
infos           = global_info()
sym_type        = infos.sym_type # sym_type
categories_id   = infos.categories_id
"""
align_rotation: find the theta component along y axis; reduce the rotation around one axis;
compute_iou(occ1, occ2): occupancy values for 3D IoU
chamfer_distance:
get_nearest_neighbors_indices_batch:

# RTs
transform_pcloud(pcloud, RT, inv=False): flexible RT/RT-1 to pcloud;
transform_points(points, transform): RT, or R
point_rotate_about_axis(pts, anchor, unitvec, theta): rotate pts around an arbitrary axis

# get R/T
rotate_pts: rotation matrix between two point sets
scale_pts: scale difference between two point sets
transform_pts(source, target): get R, T, S between two sets
rotate_about_axis(theta, axis='x'): return rotation mat around any axis;

# box & pts
get_3d_bbox(scale, shift = 0)
pts_inside_box(pts, bbox)
point_in_hull_slow(point, hull, tolerance=1e-12)
iou_3d(bbox1, bbox2, nres=50)

# camera
calculate_2d_projections(coordinates_3d, intrinsics):
calculate_3d_backprojections(depth, K, height=480, width=640, verbose=False):
project3d(pcloud_target, projMat, height=512, width=512):
project_to_camera(points, transform):

# compare RT
compute_RT_distances
axis_diff_degree
rot_diff_degree
rot_diff_rad

# mutual transform
mat_from_rvec
mat_from_euler
mat_from_quat
spherical_to_vector

# discretization
voxelize()
make_3d_grid()
normalize_coordinate(p, padding=0.1, plane='xz'):
normalize_3d_coordinate(p, padding=0.1):
normalize_coord(p, vol_range, plane='xz'):
coordinate2index(x, reso, coord_type='2d'):
coord2index(p, vol_range, reso=None, plane='xz'):
"""

def bp():
    import pdb;pdb.set_trace()

# to align the transformation,
# 0. transform the symmetric axis from (x, y, z) -> y
# 1. be able to determine the 3 rotation angle; theta_1
# 2. find the best smallest matching angle; theta_2 = theta_1 - k * interval
# 3. get the correction angle, theta_3 = theta_1 - theta_2
# 4. get the correction matrix, M = f(-theta_3)
def align_rotation(sRT, axis='y'):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    if sRT.shape[0] == 4:
        aligned_sRT[:3, 3] = sRT[:3, 3]
    return aligned_sRT

def acos_safe(x, eps=1e-4):
    sign = torch.sign(x)
    slope = np.arccos(1-eps) / eps
    return torch.where(abs(x) <= 1-eps,
                    torch.acos(x),
                    torch.acos(sign * (1 - eps)) - slope*sign*(abs(x) - 1 + eps))


def pairwise_distance_matrix(x, y, eps=1e-6):
    M, N = x.size(0), y.size(0)
    x2 = torch.sum(x * x, dim=1, keepdim=True).repeat(1, N)
    y2 = torch.sum(y * y, dim=1, keepdim=True).repeat(1, M)
    dist2 = x2 + torch.t(y2) - 2.0 * torch.matmul(x, torch.t(y))
    dist2 = torch.clamp(dist2, min=eps)
    return torch.sqrt(dist2)

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def get_cube_points(batch_size, edge_num, x_min, y_min, z_min, x_max, y_max, z_max, device):
    b = batch_size
    n = edge_num
    x_min = x_min.view(b, 1, 1).repeat((1, n * n * n, 1))
    y_min = y_min.view(b, 1, 1).repeat((1, n * n * n, 1))
    z_min = z_min.view(b, 1, 1).repeat((1, n * n * n, 1))
    x_max = x_max.view(b, 1, 1).repeat((1, n * n * n, 1))
    y_max = y_max.view(b, 1, 1).repeat((1, n * n * n, 1))
    z_max = z_max.view(b, 1, 1).repeat((1, n * n * n, 1))
    x = torch.arange(n).unsqueeze(1).repeat((1, n * n)).view(-1, 1).unsqueeze(0).repeat((b, 1, 1)).to(device)
    x = (x.float() + 1.0) / (n + 1) * (x_max - x_min) + x_min
    y = torch.arange(n).unsqueeze(1).repeat((1, n)).repeat((n, 1)).view(-1, 1).unsqueeze(0).repeat((b, 1, 1)).to(device)
    y = (y.float() + 1.0) / (n + 1) * (y_max - y_min) + y_min
    z = torch.arange(n).unsqueeze(1).repeat((n * n, 1)).view(-1, 1).unsqueeze(0).repeat((b, 1, 1)).to(device)
    z = (z.float() + 1.0) / (n + 1) * (z_max - z_min) + z_min
    cube = torch.cat((x, y, z), 2)
    return cube


def get_lattice_points(batch_size, point_num, group=1):
    n = int(np.sqrt(point_num))
    m = int(np.sqrt(group))

    x = torch.arange(n).unsqueeze(1).repeat((1, m)).view((-1, m ** 2)).repeat((1, n // m)).view((n * n, 1))
    y = torch.arange(n).view((-1, m)).repeat((1, m)).view((-1, m ** 2)).repeat((n // m, 1)).view((n * n, 1))
    patch = torch.cat((x, y), 1).unsqueeze(0).repeat((batch_size, 1, 1)).float() * (1 / n) + (0.5 / n)

    return patch


def get_lattice_grid(batch_size, point_num_x, point_num_y, delta_x=0, delta_y=0):
    a = (torch.arange(0.0, point_num_x)+delta_x)/point_num_x
    x = a.view(1, 1, point_num_x).repeat(1, point_num_y, 1)
    b = (torch.arange(0.0, point_num_y)+delta_y)/point_num_y
    y = b.view(1, point_num_y, 1).repeat(1, 1, point_num_x)

    grid = torch.cat((x, y), dim=0).unsqueeze(0)
    grid = grid.repeat(batch_size, 1, 1, 1)

    return grid

def get_spherical_lattice_grid(batch_size, point_num_x, point_num_y, delta_x=0, delta_y=0.5):
    azimuth = 2*np.pi*(torch.arange(0.0, point_num_x)+delta_x)/point_num_x
    azimuth = azimuth.view(1, 1, point_num_x).repeat(1, point_num_y, 1)

    inclination = np.pi*(torch.arange(0.0, point_num_y)+delta_y)/point_num_y
    inclination = inclination.view(1, point_num_y, 1).repeat(1, 1, point_num_x)

    x = torch.sin(inclination)*torch.cos(azimuth)
    y = torch.sin(inclination)*torch.sin(azimuth)
    z = torch.cos(inclination)

    grid = torch.cat((x, y, z), dim=0).unsqueeze(0)# (1, 3, point_num_y, point_num_x)
    grid = grid.repeat(batch_size, 1, 1, 1)

    return grid


def get_random_spherical_points(batch_size, point_num):
    pcs = torch.randn((batch_size, 3, point_num), dtype=torch.float32)
    pcs = pcs/(torch.norm(pcs, dim=1, keepdim=True) + 1e-7)
    return pcs

def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
    ''' Returns the chamfer distance for the sets of points.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        use_kdtree (bool): whether to use a kdtree
        give_id (bool): whether to return the IDs of nearest points
    '''
    if use_kdtree:
        return chamfer_distance_kdtree(points1, points2, give_id=give_id)
    else:
        return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
    '''
    assert(points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer

def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def compose_rt(rotation, translation, s=None):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation.transpose()
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    if s is not None:
        aligned_RT[:3, :3] = aligned_RT[:3, :3] * s
    return aligned_RT

def transform_pcloud(pcloud, RT, inv=False, extra_R=None, verbose=False):
    """
    by default, pcloud: [N, 3]
    """
    # if extra_R is not None:
    #     pcloud = np.dot(pcloud, extra_R[:3, :3].T)
    if RT.shape[1] == 3:
        T = 0
    else:
        T = RT[:3, 3].reshape(1, 3)
    if inv:
        inv_R     = np.linalg.pinv(RT[:3, :3])
        pcloud_tf = np.dot(inv_R, (pcloud-T).T)
        pcloud_tf = pcloud_tf.T
    else:
        pcloud_tf = np.dot(pcloud, RT[:3, :3].T) + T
    if verbose:
        print_group([RT, pcloud[:3, :], pcloud_tf[:3, :]], ['RT', 'original pts', 'transformed pts'])
        plot3d_pts([[pcloud, pcloud_tf]], [['pts', 'transformed pts']], s=1, mode='continuous',  limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]], title_name=['Camera + World Pts'])

    return pcloud_tf

def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    # Rt is B x 3 x 4
    # loc is B x 3 and scale is B
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert(Rt_new.size() == (batch_size, 3, 4))
    return Rt_new

def spherical_to_vector(spherical):
    """
    Copied from trimesh great library !
    see https://github.com/mikedh/trimesh/blob/4c9ab1e9906acaece421f
    b189437c8f4947a9c5a/trimesh/util.py
    Convert a set of (n,2) spherical vectors to (n,3) vectors
    Parameters
    -----------
    spherical : (n , 2) float
       Angles, in radians
    Returns
    -----------
    vectors : (n, 3) float
      Unit vectors
    """
    spherical = np.asanyarray(spherical, dtype=np.float64)

    theta, phi = spherical.T
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    vectors = np.column_stack((ct * sp, st * sp, cp))
    return vectors

def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

def normalize_coord(p, vol_range, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments

    Args:
        p (tensor): point
        vol_range (numpy array): volume boundary
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    p[:, 0] = (p[:, 0] - vol_range[0][0]) / (vol_range[1][0] - vol_range[0][0])
    p[:, 1] = (p[:, 1] - vol_range[0][1]) / (vol_range[1][1] - vol_range[0][1])
    p[:, 2] = (p[:, 2] - vol_range[0][2]) / (vol_range[1][2] - vol_range[0][2])

    if plane == 'xz':
        x = p[:, [0, 2]]
    elif plane =='xy':
        x = p[:, [0, 1]]
    elif plane =='yz':
        x = p[:, [1, 2]]
    else:
        x = p
    return x

def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def coord2index(p, vol_range, reso=None, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): points
        vol_range (numpy array): volume boundary
        reso (int): defined resolution
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    # normalize to [0, 1]
    x = normalize_coord(p, vol_range, plane=plane)

    if isinstance(x, np.ndarray):
        x = np.floor(x * reso).astype(int)
    else: #* pytorch tensor
        x = (x * reso).long()

    if x.shape[1] == 2:
        index = x[:, 0] + reso * x[:, 1]
        index[index > reso**2] = reso**2
    elif x.shape[1] == 3:
        index = x[:, 0] + reso * (x[:, 1] + reso * x[:, 2])
        index[index > reso**3] = reso**3

    return index[None]

def update_reso(reso, depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        reso (int): defined resolution
        depth (int): U-Net number of layers
    '''
    base = 2**(int(depth) - 1)
    if ~(reso / base).is_integer(): # when this is not integer, U-Net dimension error
        for i in range(base):
            if ((reso + i) / base).is_integer():
                reso = reso + i
                break
    return reso

def decide_total_volume_range(query_vol_metric, recep_field, unit_size, unet_depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        query_vol_metric (numpy array): query volume size
        recep_field (int): defined the receptive field for U-Net
        unit_size (float): the defined voxel size
        unet_depth (int): U-Net number of layers
    '''
    reso = query_vol_metric / unit_size + recep_field - 1
    reso = update_reso(int(reso), unet_depth) # make sure input reso can be processed by UNet
    input_vol_metric = reso * unit_size
    p_c = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    lb_input_vol, ub_input_vol = p_c - input_vol_metric/2, p_c + input_vol_metric/2
    lb_query_vol, ub_query_vol = p_c - query_vol_metric/2, p_c + query_vol_metric/2
    input_vol = [lb_input_vol, ub_input_vol]
    query_vol = [lb_query_vol, ub_query_vol]

    # handle the case when resolution is too large
    if reso > 10000:
        reso = 1

    return input_vol, query_vol, reso

def add_key(base, new, base_name, new_name, device=None):
    ''' Add new keys to the given input

    Args:
        base (tensor): inputs
        new (tensor): new info for the inputs
        base_name (str): name for the input
        new_name (str): name for the new info
        device (device): pytorch device
    '''
    if (new is not None) and (isinstance(new, dict)):
        if device is not None:
            for key in new.keys():
                new[key] = new[key].to(device)
        base = {base_name: base,
                new_name: new}
    return base

class map2local(object):
    ''' Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    '''
    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s # always possitive
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!
        p = self.pe(p)
        return p

class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0 # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p

def mat_from_rvec(rvec):
    angle = np.linalg.norm(rvec)
    axis = np.array(rvec).reshape(3) / angle if angle != 0 else [0, 0, 1]
    mat = tf.axangles.axangle2mat(axis, angle)
    return np.matrix(mat)


def mat_from_euler(ai, aj, ak, axes='sxyz'):
    mat = tf.euler.euler2mat(ai, aj, ak, axes)
    return np.matrix(mat)


def mat_from_quat(quat):
    x, y, z, w = quat
    mat = tf.quaternions.quat2mat((w, x, y, z))
    return np.matrix(mat)


def rvec_from_mat(mat):
    axis, angle = tf.axangles.mat2axangle(mat, unit_thresh=1e-03)
    rvec = axis * angle
    return rvec


def rvec_from_quat(quat):
    x, y, z, w = quat
    axis, angle = tf.quaternions.quat2axangle((w, x, y, z),
                                              identity_thresh=1e-06)
    rvec = axis * angle
    return rvec


def quat_from_mat(mat):
    w, x, y, z = tf.quaternions.mat2quat(mat)
    return (x, y, z, w)


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3

    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )

    theta = torch.acos(cos)

    #theta = torch.min(theta, 2*np.pi - theta)


    return theta

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to pi batch
def compute_angle_from_r_matrices(m):

    batch=m.shape[0]

    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )

    theta = torch.acos(cos)

    return theta


#euler batch*3
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]

    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1

    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3

    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3

    return matrix

#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()

    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)
    z=torch.atan2(R[:,1,0],R[:,0,0])

    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)
    zs=R[:,1,0]*0

    out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular

    return out_euler

# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

# # from zhou et. al
# def compute_rotation_matrix_from_ortho6d(poses):
#     x_raw = poses[:,0:3]#batch*3
#     y_raw = poses[:,3:6]#batch*3
#
#     x = normalize_vector(x_raw) #batch*3
#     z = cross_product(x,y_raw) #batch*3
#     z = normalize_vector(z)#batch*3
#     y = cross_product(z,x)#batch*3
#
#     x = x.view(-1,3,1)
#     y = y.view(-1,3,1)
#     z = z.view(-1,3,1)
#     matrix = torch.cat((x,y,z), 2) #batch*3*3
#
#     return matrix

# # batch*n
# def normalize_vector( v):
#     batch=v.shape[0]
#     v_mag = torch.sqrt(v.pow(2).sum(1))# batch
#     v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
#     v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
#     v = v/v_mag
#     return v

# batch*n
def normalize_vector(v):  # v: [B, 3]
    batch = v.shape[0]
    v_mag = torch.norm(v, p=2, dim=1)  # batch
    """
    if torch.any(torch.isnan(v_mag)):
        print('nan in v_mag!')
        idx = torch.where(torch.isnan(v_mag))[0]
        print('v_mag', v_mag[idx], 'v', v[idx])
    if torch.any(torch.isinf(v_mag)):
        print('inf in v_mag!')
        idx = torch.where(torch.isinf(v_mag))[0]
        print('v_mag', v_mag[idx], 'v', v[idx])
    """
    eps = torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v_mag.device))
    valid_mask = (v_mag > eps).float().view(batch, 1)
    backup = torch.tensor([1.0, 0.0, 0.0]).float().to(v.device).view(1, 3).expand(batch, 3)
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    ret = v * valid_mask + backup * (1 - valid_mask)
    """
    if torch.any(torch.isnan(v)):
        print('nan in v!')
        idx = torch.where(torch.isnan(v))[0]
        print('v_mag', v_mag[idx], 'v', v[idx], 'valid', valid_mask[idx])
    if torch.any(torch.isnan(ret)):
        print('nan in ret!')
        idx = torch.where(torch.isnan(ret))[0]
        print('v_mag', v_mag[idx], 'v', v[idx], 'valid', valid_mask[idx])
    """
    return ret

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3

    return out

# from He et. al
def get_3d_bbox(scale, shift = 0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - np.reshape(bbox[4, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1>0, p1<np.dot(u1, u1))
    p2 = np.logical_and(p2>0, p2<np.dot(u2, u2))
    p3 = np.logical_and(p3>0, p3<np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)

def point_in_hull_slow(point, hull, tolerance=1e-12):
    """
    Check if a point lies in a convex hull.
    :param point: nd.array (1 x d); d: point dimension
    :param hull: The scipy ConvexHull object
    :param tolerance: Used to compare to a small positive constant because of issues of numerical precision
    (otherwise, you may find that a vertex of the convex hull is not in the convex hull)
    """
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

def iou_3d(bbox1, bbox2, nres=50):
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.linspace(bmin[0], bmax[0], nres)
    ys = np.linspace(bmin[1], bmax[1], nres)
    zs = np.linspace(bmin[2], bmax[2], nres)
    pts = np.array([x for x in itertools.product(xs, ys, zs)])
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union==0:
        return 1
    else:
        return intersect/float(union)

# def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
#     """
#     This method compares predictions to the ground truth in terms of scale.
#     It is equivalent to intersection over union (IOU) between the two boxes in 3D,
#     if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
#     :param sample_annotation: GT annotation sample.
#     :param sample_result: Predicted sample.
#     :return: Scale IOU.
#     """
#     # Validate inputs.
#     sa_size = np.array(sample_annotation.size)
#     sr_size = np.array(sample_result.size)
#     assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
#     assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

#     # Compute IOU.
#     min_wlh = np.minimum(sa_size, sr_size)
#     volume_annotation = np.prod(sa_size)
#     volume_result = np.prod(sr_size)
#     intersection = np.prod(min_wlh)  # type: float
#     union = volume_annotation + volume_result - intersection  # type: float
#     iou = intersection / union

#     return iou

def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    if coordinates.shape[0] != 3 and coordinates.shape[1]==3:
        # print('transpose box channels')
        coordinates = coordinates.transpose()
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_3d_backprojections(depth, K, height=480, width=640, verbose=False):
    # backproject to camera space
    xmap = np.array([[j for i in range(width)] for j in range(height)])
    ymap = np.array([[i for i in range(width)] for j in range(height)])
    cam_cx = K[0, 2]
    cam_cy = K[1, 2]
    cam_fx = K[0, 0]
    cam_fy = K[1, 1]
    # cam_fx = K[0, 0]
    # cam_fy = K[1, 1]
    cam_scale = 1
    pt2 = depth / cam_scale
    pt0 = (ymap - cam_cx) * pt2 / cam_fx
    pt1 = (xmap - cam_cy) * pt2 / cam_fy
    cloud = np.stack((pt0, pt1, pt2), axis=-1)
    if verbose:
        import __init__
        from common.vis_utils import plot3d_pts
        plot3d_pts([[cloud.reshape(-1, 3)]], [['Part {}'.format(j) for j in range(1)]], s=3**2, title_name=['cam pc'], sub_name=str(0), axis_off=False, save_fig=False)

    return cloud

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input:
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def compute_RT_distances(RT_1, RT_2):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    '''
    #print(RT_1[3, :], RT_2[3, :])
    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1

    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])

    R1 = RT_1[:3, :3]/np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3]/np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
    shift = np.linalg.norm(T1-T2) * 100
    # print(theta, shift)

    if theta < 5 and shift < 5:
        return 10 - theta - shift
    else:
        return -1

def axis_diff_degree(v1, v2, category=''):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    r_diff = np.arccos(np.sum(v1*v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
    if category in sym_type and len(sym_type[category].keys()) > 1: # when we have x or z sym besides y
        print(f'for {category}, tolerate upside down')
        return min(r_diff, 180-r_diff)
    else:
        return np.abs(r_diff)

def rot_diff_degree(rot1, rot2, up=False):
    if up:
        y_axis = np.array([[0, 1, 0]])
        tv1 = transform_pcloud(y_axis, rot1)
        tv2 = transform_pcloud(y_axis, rot2)
        return tv1, tv2, axis_diff_degree(tv1, tv2)
    else:
        y_axis = np.array([[0, 1, 0]])
        tv1 = transform_pcloud(y_axis, rot1)
        tv2 = transform_pcloud(y_axis, rot2)
        return tv1, tv2, rot_diff_rad(rot1, rot2) / np.pi * 180

def rot_diff_rad(rot1, rot2, category=''):
    # default rot2 is gt
    if category in sym_type and len(sym_type[category].keys()) > 0:
        all_rmats = [np.eye(3)]
        for key, M in sym_type[category].items():
            next_rmats = []
            for k in range(M):
                rmat = rotate_about_axis(2 * np.pi * k / M, axis=key)
                for old_rmat in all_rmats:
                    next_rmats.append(np.matmul(rmat, old_rmat))
            all_rmats = next_rmats
        all_rmats = np.stack(all_rmats, axis=0) # N, 3, 3
        rot2_group= np.matmul(all_rmats.transpose(0, 2, 1), rot2) # we need P R^T * r^T
        return min(np.arccos( ( np.trace(np.matmul(rot2_group, rot1.T), axis1=1, axis2=2) - 1 ) / 2 ) % (2*np.pi))
    else:
        return np.arccos( ( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2 ) % (2*np.pi)

def rotate_points_with_rotvec(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def dist_between_3d_lines(p1, e1, p2, e2):
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    product = np.sum(orth_vect * (p1 - p2))
    dist = product / np.linalg.norm(orth_vect)

    return np.abs(dist)

def project3d(pcloud_target, projMat, height=512, width=512):
    pcloud_projected = np.dot(pcloud_target, projMat.T)
    pcloud_projected_ndc = pcloud_projected/pcloud_projected[:, 3:4]
    img_coord = (pcloud_projected_ndc[:, 0:2] + 1)/(1/256)
    print('transformed image coordinates:\n', img_coord.shape)
    u = img_coord[:, 0]
    v = img_coord[:, 1]
    u = u.astype(np.int16)
    v = v.astype(np.int16)
    v = 512 - v
    print('u0, v0:\n', u[0], v[0])

    return u, v # x, y in cv coords


def point_3d_offset_joint(joint, point):
    """
    joint: [x, y, z] or [[x, y, z] + [rx, ry, rz]]
    point: N * 3
    """
    if len(joint) == 2:
        P0 = np.array(joint[0])
        P  = np.array(point)
        l  = np.array(joint[1]).reshape(1, 3)
        P0P= P - P0
        PP = np.dot(P0P, l.T) * l / np.linalg.norm(l)**2  - P0P
    return PP


def rotate_pts(source, target):
    '''
    func: compute rotation between source: [N x 3], target: [N x 3]
    '''
    # pre-centering
    source = source - np.mean(source, 0, keepdims=True)
    target = target - np.mean(target, 0, keepdims=True)
    M = np.matmul(target.T, source)
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    R = np.matmul(U, Vh)
    return R


def transform_pts(source, target):
    # source: [N x 3], target: [N x 3]
    # pre-centering and compute rotation
    source_centered = source - np.mean(source, 0, keepdims=True)
    target_centered = target - np.mean(target, 0, keepdims=True)
    rotation = rotate_pts(source_centered, target_centered)

    scale = scale_pts(source_centered, target_centered)

    # compute translation
    translation = np.mean(target.T-scale*np.matmul(rotation, source.T), 1)
    return rotation, scale, translation


def scale_pts(source, target):
    '''
    func: compute scaling factor between source: [N x 3], target: [N x 3]
    '''
    pdist_s = source.reshape(source.shape[0], 1, 3) - source.reshape(1, source.shape[0], 3)
    A = np.sqrt(np.sum(pdist_s**2, 2)).reshape(-1)
    pdist_t = target.reshape(target.shape[0], 1, 3) - target.reshape(1, target.shape[0], 3)
    b = np.sqrt(np.sum(pdist_t**2, 2)).reshape(-1)
    scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    return scale

def compute_3d_rotation_axis(pts_0, pts_1, rt, orientation=None, line_pts=None, methods='H-L', item='eyeglasses', viz=False):
    """
    pts_0: points in NOCS space of cannonical status(scaled)
    pts_1: points in camera space retrieved from depth image;
    rt: rotation + translation in 4 * 4
    """
    num_parts = len(rt)
    print('we have {} parts'.format(num_parts))

    chained_pts = [None] * num_parts
    delta_Ps = [None] * num_parts
    chained_pts[0] = np.dot( np.concatenate([ pts_0[0], np.ones((pts_0[0].shape[0], 1)) ], axis=1), rt[0].T )
    axis_list = []
    angle_list= []
    if item == 'eyeglasses':
        for j in range(1, num_parts):
            chained_pts[j] = np.dot(np.concatenate([ pts_0[j], np.ones((pts_0[j].shape[0], 1)) ], axis=1), rt[0].T)

            if methods == 'H-L':
                RandIdx = np.random.randint(chained_pts[j].shape[1], size=5)
                orient, position= estimate_joint_HL(chained_pts[j][RandIdx, 0:3], pts_1[j][RandIdx, 0:3])
                joint_axis = {}
                joint_axis['orient']   = orient
                joint_axis['position'] = position
                source_offset_arr= point_3d_offset_joint([position.reshape(1, 3), orient], chained_pts[j][RandIdx, 0:3])
                rotated_offset_arr= point_3d_offset_joint([position.reshape(1, 3), orient.reshape(1, 3)], pts_1[j][RandIdx, 0:3])
                angle = []
                for m in range(RandIdx.shape[0]):
                    modulus_0 = np.linalg.norm(source_offset_arr[m, :])
                    modulus_1 = np.linalg.norm(rotated_offset_arr[m, :])
                    cos_angle = np.dot(source_offset_arr[m, :].reshape(1, 3), rotated_offset_arr[m, :].reshape(3, 1))/(modulus_0 * modulus_1)
                    angle_per_pair = np.arccos(cos_angle)
                    angle.append(angle_per_pair)
                print('angle per pair from multiple pairs: {}', angle)
                angle_list.append(sum(angle)/len(angle))

            axis_list.append(joint_axis)
            angle_list.append(angle)

    return axis_list, angle_list

def point_rotate_about_axis(pts, anchor, unitvec, theta):
    a, b, c = anchor.reshape(3)
    u, v, w = unitvec.reshape(3)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    ss =  u*x + v*y + w*z
    x_rotated = (a*(v**2 + w**2) - u*(b*v + c*w - ss)) * (1 - cos(theta)) + x * cos(theta) + (-c*v + b*w - w*y + v*z) * sin(theta)
    y_rotated = (b*(u**2 + w**2) - v*(a*u + c*w - ss)) * (1 - cos(theta)) + y * cos(theta) + (c*u - a*w + w*x - u*z) * sin(theta)
    z_rotated = (c*(u**2 + v**2) - w*(a*u + b*v - ss)) * (1 - cos(theta)) + z * cos(theta) + (-b*u + a*v - v*x + u*y) * sin(theta)
    rotated_pts = np.zeros_like(pts)
    rotated_pts[:, 0] = x_rotated
    rotated_pts[:, 1] = y_rotated
    rotated_pts[:, 2] = z_rotated

    return rotated_pts

def rotate_eular(theta_x, theta_y, theta_z):
    Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
    Ry = rotate_about_axis(theta_x / 180 * np.pi, axis='y')
    Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
    r = Rz @ Ry @ Rx

    return r

def rotate_about_axis(theta, axis='x'):
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, cos(theta), -sin(theta)],
                      [0, sin(theta), cos(theta)]])

    elif axis == 'y':
        R = np.array([[cos(theta), 0, sin(theta)],
                      [0, 1, 0],
                      [-sin(theta), 0, cos(theta)]])

    elif axis == 'z':
        R = np.array([[cos(theta), -sin(theta), 0],
                      [sin(theta), cos(theta),  0],
                      [0, 0, 1]])
    return R
# def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2*np.pi) -> float:
#     """
#     Returns the yaw angle difference between the orientation of two boxes.
#     :param gt_box: Ground truth box.
#     :param eval_box: Predicted box.
#     :param period: Periodicity in radians for assessing angle difference.
#     :return: Yaw angle difference in radians in [0, pi].
#     """
#     yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
#     yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

#     return abs(angle_diff(yaw_gt, yaw_est, period))


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def estimate_joint_HL(source_pts, rotated_pts):
    # estimate offsets
    delta_P = rotated_pts - source_pts
    assert delta_P.shape[1] == 3, 'points coordinates dimension is wrong, current is {}'.format(delta_P.shape)
    mid_pts = (source_pts + rotated_pts)/2
    CC      = np.zeros((3, 3), dtype=np.float32)
    BB      = np.zeros((delta_P.shape[0], 1), dtype=np.float32)
    for j in range(0, delta_P.shape[0]):
        CC += np.dot(delta_P[j, :].reshape(3, 1), delta_P[j, :].reshape(1, 3))
        BB[j] = np.dot(delta_P[j, :].reshape(1, 3), mid_pts[j, :].reshape((3, 1)))
    w, v   = np.linalg.eig(CC)
    print('eigen vectors are: \n', v)
    print('eigne values are: \n', w)
    orient = v[:, np.argmin(np.squeeze(w))].reshape(3, 1)

    # we already decouple the orient & position
    mat_1 = np.linalg.pinv( np.dot(delta_P.T, delta_P) )

    position = np.dot( np.dot(mat_1, delta_P.T), BB)
    print('orient has shape {}, position has shape {}'.format(orient.shape, position.shape))

    return orient, position


# def calc_displace_vector(points: np.array, curr_box: Box, next_box: Box):
#     """
#     Calculate the displacement vectors for the input points.
#     This is achieved by comparing the current and next bounding boxes. Specifically, we first rotate
#     the input points according to the delta rotation angle, and then translate them. Finally we compute the
#     displacement between the transformed points and the input points.
#     :param points: The input points, (N x d). Note that these points should be inside the current bounding box.
#     :param curr_box: Current bounding box.
#     :param next_box: The future next bounding box in the temporal sequence.
#     :return: Displacement vectors for the points.
#     """
#     assert points.shape[1] == 3, "The input points should have dimension 3."

#     # Make sure the quaternions are normalized
#     curr_box.orientation = curr_box.orientation.normalised
#     next_box.orientation = next_box.orientation.normalised

#     delta_rotation = curr_box.orientation.inverse * next_box.orientation
#     rotated_pc = (delta_rotation.rotation_matrix @ points.T).T
#     rotated_curr_center = np.dot(delta_rotation.rotation_matrix, curr_box.center)
#     delta_center = next_box.center - rotated_curr_center

#     rotated_tranlated_pc = rotated_pc + delta_center

#     pc_displace_vectors = rotated_tranlated_pc - points

#     return pc_displace_vectors

def voxelize(pts, voxel_size, extents=None, num_T=35, seed: float = None):
    """
    Voxelize the input point cloud. Code modified from https://github.com/Yc174/voxelnet
    Voxels are 3D grids that represent occupancy info.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be -1 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param num_T: Number of points in each voxel after sampling/padding
    :param seed: The random seed for fixing the data generation.
    """
    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    points = pts[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Number of points per voxel, last voxel calculated separately
    out_points_in_voxel = np.diff(unique_indices)
    out_points_in_voxel = np.append(out_points_in_voxel, discrete_pts.shape[0] - unique_indices[-1])

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Padding the points within each voxel
    padded_voxel_points = np.zeros([unique_indices.shape[0], num_T, pts.shape[1] + 3], dtype=np.float32)
    padded_voxel_points = padding_voxel(padded_voxel_points, unique_indices, out_points_in_voxel, points, num_T, seed)

    return padded_voxel_points, voxel_indices, num_divisions

if __name__ == '__main__':
    #>>>>>>>>> 3D IOU compuatation
    from scipy.spatial.transform import Rotation
    bbox1 = np.array([[-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]])
    print('bbox1.shape: ', bbox1.shape)
    rotmatrix = Rotation.from_rotvec(np.pi/4 * np.array([np.sqrt(3)/3, np.sqrt(3)/3, np.sqrt(3)/3])).as_dcm()
    bbox2 = np.matmul(bbox1, rotmatrix.T)
    bbox3 = bbox1 + np.array([[1, 0, 0]])
    rotmatrix2 = Rotation.from_rotvec(np.pi/4 * np.array([0, 0, 1])).as_dcm()
    bbox4 = np.matmul(bbox1, rotmatrix2.T)
    bbox5 = bbox1 + np.array([[2, 0, 0]])
    print(iou_3d(bbox1, bbox1))
    print(iou_3d(bbox1, bbox2))
    print(iou_3d(bbox1, bbox3))
    print(iou_3d(bbox1, bbox4))
    print(iou_3d(bbox1, bbox5))
    #>>>>>>>>> test for joint parameters fitting
    source_pts  = np.array([[5, 1, 5], [0, 0, 1], [0.5,0.5,0.5], [2, 0, 1], [3, 3, 5]])
    p1 = np.array([0,0,0])
    p2 = np.array([1,1,1])
    unitvec = (p2 - p1) / np.linalg.norm(p2 - p1)
    anchor  = p1
    rotated_pts = point_rotate_about_axis(source_pts, anchor, unitvec, pi)
    joint_axis, position = estimate_joint_HL(source_pts, rotated_pts)
    print(joint_axis, position)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], c='r',
               marker='o', label='source pts')
    ax.scatter(rotated_pts[:, 0], rotated_pts[:, 1], rotated_pts[:, 2], c='b',
               marker='o', label='rotated pts')
    linepts = unitvec * np.mgrid[-5:5:2j][:, np.newaxis] + np.array(p1).reshape(1, 3)
    ax.plot3D(*linepts.T, linewidth=5, c='green')
    ax.legend(loc='lower left')
    plt.show()
