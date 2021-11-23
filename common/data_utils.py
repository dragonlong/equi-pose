import numpy as np
import random
import os
import time
import h5py
import csv
import pickle
import yaml
import json
import re
import os.path
import sys
import struct
import trimesh
# from numba import njit
from numpy.linalg import inv
from pytransform3d.rotations import *
import cv2
from plyfile import PlyData, PlyElement

from io import BytesIO

import glob
import platform
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
from oiio import OpenImageIO as oiio

import __init__
from common.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from common.vis_utils import plot_arrows, plot_arrows_list, plot_arrows_list_threshold, plot3d_pts, plot_lines, plot_hand_w_object
from global_info import global_info
import torchvision

mc_client = None

"""
mesh: fast_load_obj, get_obj_mesh(R correct)
urdf: get_urdf
nocs: calculate_factor_nocs
"""

infos = global_info()
epsilon = 10e-8
second_path = infos.second_path
render_path = infos.render_path
viz_path  = infos.viz_path
whole_obj = infos.whole_obj

def breakpoint():
    import pdb;pdb.set_trace()


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.png', '.jpg']:
            return cls._read_img(file_path)
        elif file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.exr']:
            return cls._read_exr(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.pkl']:
            return cls._read_pkl(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def put(cls, file_path, file_content):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.pcd']:
            return cls._write_pcd(file_path, file_content)
        elif file_extension in ['.h5']:
            return cls._write_h5(file_path, file_content)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_img(cls, file_path):
        if mc_client is None:
            return cv2.imread(file_path, cv2.IMREAD_UNCHANGED) / 255.
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            img_array = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            return img / 255.

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        if mc_client is None:
            return np.load(file_path)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            buf_bytes = buf.tobytes()
            if not buf_bytes[:6] == b'\x93NUMPY':
                raise Exception('Invalid npy file format.')

            header_size = int.from_bytes(buf_bytes[8:10], byteorder='little')
            header = eval(buf_bytes[10:header_size + 10])
            dtype = np.dtype(header['descr'])
            nd_array = np.frombuffer(buf[header_size + 10:], dtype).reshape(header['shape'])

            return nd_array

    def _read_pkl(cls, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        return data
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        if mc_client is None:
            pc = open3d.io.read_point_cloud(file_path)
            ptcloud = np.array(pc.points)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            text = mc.ConvertString(pyvector).split('\n')
            start_line_idx = len(text) - 1
            for idx, line in enumerate(text):
                if line == 'DATA ascii':
                    start_line_idx = idx + 1
                    break

            ptcloud = text[start_line_idx:]
            ptcloud = np.genfromtxt(BytesIO('\n'.join(ptcloud).encode()), dtype=np.float32)

        # ptcloud = np.concatenate((ptcloud, np.array([[0, 0, 0]])), axis=0)
        return ptcloud

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        # Avoid overflow while gridding
        return f['data'][()] * 0.9

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _write_pcd(cls, file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc)

    def _read_exr(cls, filename):
        """
        Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
        """
        inbuf=oiio.ImageInput.open(filename)
        img  = inbuf.read_image()
        Z = None
        inbuf.close()
        return img, Z

    @classmethod
    def _write_h5(cls, file_path, file_content):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=file_content)


def read_ply_xyz(filename):
    """ read XYZ point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices


def write_obj(points, faces, filename):
    with open(filename, 'w') as F:
        for p in points:
            F.write('v %f %f %f\n'%(p[0], p[1], p[2]))

        for f in faces:
            F.write('f %d %d %d\n'%(f[0], f[1], f[2]))


def convert_point_cloud_to_balls(pc_ply_filename):
    if not pc_ply_filename.endswith('.ply'):
        return

    shape_id = pc_ply_filename.split('/')[-2]
    ply_name = pc_ply_filename.split('/')[-1]
    if 'raw.ply' == ply_name:
        sphere_r = 0.012
    else:
        if 'airplane' in pc_ply_filename:
            sphere_r = 0.012
        else:
            sphere_r = 0.016

    output_path = os.path.dirname(os.path.dirname(pc_ply_filename)) + '_spheres'
    out_shape_dir = os.path.join(output_path, shape_id)
    if not os.path.exists(out_shape_dir):
        os.makedirs(out_shape_dir)
    output_filename = os.path.join(out_shape_dir, ply_name[:-3] + '_{:.4f}.obj'.format(sphere_r))
    if os.path.exists(output_filename):
        return

    pc = read_ply_xyz(pc_ply_filename)

    points = []
    faces = []

    for pts in (pc):
        sphere_m = trimesh.creation.uv_sphere(radius=sphere_r, count=[8,8])
        sphere_m.apply_translation(pts)

        faces_offset = np.array(sphere_m.faces) + len(points)
        faces.extend(faces_offset)
        points.extend(np.array(sphere_m.vertices))

    points = np.array(points)
    faces = np.array(faces)
    #print(points.shape, faces.shape)
    finale_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    finale_mesh.export(output_filename)

#>>>>>>>>>>>>>>>>>>>>>>>>>> data augmentation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor

def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue)

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img

def get_obj_mesh(basename=None, full_path=None, category='eyeglasses', verbose=False):
    """
    basename: 0001_7_0_2
    """
    verts = []
    faces = []

    if full_path is not None:
        objname = full_path
    else:
        attrs = basename.split('_')
        instance = attrs[0]
        arti_ind = attrs[1]
        objname = f'{whole_obj}/{category}/{instance}/{arti_ind}.obj'
    obj= fast_load_obj(open(objname, 'rb'))[0] # why it is [0]
    obj_verts = obj['vertices']
    obj_faces = obj['faces']
    alpha = -np.pi/2
    correct_R = matrix_from_euler_xyz([alpha, 0, 0])

    # transfrom mesh into blender coords
    obj_verts = np.dot(correct_R, obj_verts.T).T

    verts.append(obj_verts)
    faces.append(obj_faces)
    if verbose:
        plot_hand_w_object(obj_verts=verts[0]-np.mean(verts[0], axis=0, keepdims=True), obj_faces=faces[0], hand_verts=verts[0]-np.mean(verts[0], axis=0, keepdims=True), hand_faces=faces[0],save=False, mode='continuous')

    return verts[0], faces[0]

def spherical_to_vector(spherical):
    """
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


def sample_surface_sphere(count, center=np.array([0, 0, 0]), radius=1):
    """
    http://mathworld.wolfram.com/SpherePointPicking.html
    Parameters
    ----------
    count: int, number of points to return
    Returns
    ----------
    points: (count,3) float, list of random points on a unit sphere
    """

    u, v = np.random.random((2, count))

    theta = np.pi * 2 * u
    phi = np.arccos((2 * v) - 1)

    points = center + radius * spherical_to_vector(
        np.column_stack((theta, phi)))
    return points


def sample_mesh(mesh, min_hits=2000, ray_nb=3000, interrupt=10):
    verts = np.array(mesh.vertices)
    centroid = verts.mean(0)
    radius = max(np.linalg.norm(verts - centroid, axis=1))
    # print('radius ', radius)
    origins = sample_surface_sphere(ray_nb, centroid, radius=2 * radius)
    hits = None
    counts = 0
    while hits is None or hits.shape[0] < min_hits:
        counts += 1
        destination = centroid + sample_surface_sphere(ray_nb, radius=radius)
        directions = destination - (origins)
        # print('Casting rays ! ')
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False)
        # print('Got {} hits'.format(locations.shape))
        if hits is None:
            hits = locations
        else:
            hits = np.concatenate([hits, locations])
        if counts > interrupt:
            raise Exception('Exceeded {} attempts'.format(interrupt))
    return hits
    # return hits, centroid, destination


def tri_area(v):
    return 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def points_from_mesh(faces, vertices, vertex_nb=600, show_cloud=False):
    """
    Points are sampled on the surface of the mesh, with probability
    proportional to face area
    """
    areas = tri_area(vertices[faces])

    proba = areas / areas.sum()
    rand_idxs = np.random.choice(
        range(areas.shape[0]), size=vertex_nb, p=proba)

    # Randomly pick points on triangles
    u = np.random.rand(vertex_nb, 1)
    v = np.random.rand(vertex_nb, 1)

    # Force bernouilli couple to be picked on a half square
    out = u + v > 1
    u[out] = 1 - u[out]
    v[out] = 1 - v[out]

    rand_tris = vertices[faces[rand_idxs]]
    points = rand_tris[:, 0] + u * (rand_tris[:, 1] - rand_tris[:, 0]) + v * (
        rand_tris[:, 2] - rand_tris[:, 0])

def write_pointcloud(filename,xyz_points,rgb_points=None):
    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))
    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),rgb_points[i,2].tostring())))
    fid.close()

def get_urdf_mobility(inpath, verbose=True):
    urdf_ins = {}
    tree_urdf     = ET.parse(inpath + "/mobility.urdf") # todo
    num_real_links= len(tree_urdf.findall('link'))
    root_urdf     = tree_urdf.getroot()
    rpy_xyz       = {}
    list_xyz      = [None] * num_real_links
    list_rpy      = [None] * num_real_links
    list_box      = [None] * num_real_links
    list_obj      = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    num_links     = 0
    for link in root_urdf.iter('link'):
        num_links += 1
        index_link = None
        if link.attrib['name']=='base':
            index_link = 0
        else:
            index_link = int(link.attrib['name'].split('_')[1]) + 1 # since the name is base, link_0, link_1
        list_xyz[index_link] = []
        list_rpy[index_link] = []
        list_obj[index_link] = []
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[index_link].append([float(x) for x in origin.attrib['xyz'].split()])
                else:
                    list_xyz[index_link].append([0, 0, 0])
                if 'rpy' in origin.attrib:
                    list_rpy[index_link].append([float(x) for x in origin.attrib['rpy'].split()])
                else:
                    list_rpy[index_link].append([0, 0, 0])
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    if 'home' in mesh.attrib['filename'] or 'work' in mesh.attrib['filename']:
                        list_obj[index_link].append(mesh.attrib['filename'])
                    else:
                        list_obj[index_link].append(inpath + '/' + mesh.attrib['filename'])

    rpy_xyz['xyz']   = list_xyz
    rpy_xyz['rpy']   = list_rpy # here it is empty list
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj

    rpy_xyz       = {}
    list_type     = [None] * (num_real_links - 1)
    list_parent   = [None] * (num_real_links - 1)
    list_child    = [None] * (num_real_links - 1)
    list_xyz      = [None] * (num_real_links - 1)
    list_rpy      = [None] * (num_real_links - 1)
    list_axis     = [None] * (num_real_links - 1)
    list_limit    = [[0, 0]] * (num_real_links - 1)
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        joint_index            = int(joint.attrib['name'].split('_')[1])
        list_type[joint_index] = joint.attrib['type']

        for parent in joint.iter('parent'):
            link_name = parent.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            list_parent[joint_index] = link_index
        for child in joint.iter('child'):
            link_name = child.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            list_child[joint_index] = link_index
        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                list_xyz[joint_index] = [0, 0, 0]
            if 'rpy' in origin.attrib:
                list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                list_rpy[joint_index] = [0, 0, 0]
        for axis in joint.iter('axis'): # we must have
            list_axis[joint_index]= [float(x) for x in axis.attrib['xyz'].split()]
        for limit in joint.iter('limit'):
            list_limit[joint_index]= [float(limit.attrib['lower']), float(limit.attrib['upper'])]

    rpy_xyz['type']      = list_type
    rpy_xyz['parent']    = list_parent
    rpy_xyz['child']     = list_child
    rpy_xyz['xyz']       = list_xyz
    rpy_xyz['rpy']       = list_rpy
    rpy_xyz['axis']      = list_axis
    rpy_xyz['limit']     = list_limit


    urdf_ins['joint']    = rpy_xyz
    urdf_ins['num_links']= num_real_links
    if verbose:
        for j, pos in enumerate(urdf_ins['link']['xyz']):
            if len(pos) > 3:
                print('link {} xyz: '.format(j), pos[0])
            else:
                print('link {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['link']['rpy']):
            if len(orient) > 3:
                print('link {} rpy: '.format(j), orient[0])
            else:
                print('link {} rpy: '.format(j), orient)
        # for joint
        for j, pos in enumerate(urdf_ins['joint']['xyz']):
            print('joint {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['joint']['rpy']):
            print('joint {} rpy: '.format(j), orient)
        for j, orient in enumerate(urdf_ins['joint']['axis']):
            print('joint {} axis: '.format(j), orient)
        for j, child in enumerate(urdf_ins['joint']['child']):
            print('joint {} has child link: '.format(j), child)
        for j, parent in enumerate(urdf_ins['joint']['parent']):
            print('joint {} has parent link: '.format(j), parent)
        # plot_lines(urdf_ins['joint']['axis'])

    return urdf_ins

def get_boundary(cpts):
    p = 0
    x_min = cpts[p][0][0][0]
    y_min = cpts[p][0][0][1]
    z_min = cpts[p][0][0][2]
    x_max = cpts[p][1][0][0]
    y_max = cpts[p][1][0][1]
    z_max = cpts[p][1][0][2]
    boundary = np.array([[x_min, y_min, z_min],
                           [x_max, y_min, z_min],
                           [x_max, y_min, z_max],
                           [x_min, y_min, z_max],
                           [x_min, y_max, z_min],
                           [x_min, y_max, z_max],
                           [x_max, y_max, z_max],
                          [x_max, y_max, z_min],
                          [x_min, y_min, z_max],
                          [x_min, y_max, z_max],
                          [x_max, y_max, z_max],
                          [x_max, y_min, z_max],
                          ]
                          )
    return boundary

def load_model_split(inpath):
    nsplit = []
    tsplit = []
    vsplit = []
    fsplit = []
    vcount = 0
    vncount = 0
    vtcount = 0
    fcount = 0
    dict_mesh = {}
    list_group= []
    list_xyz  = []
    list_face = []
    list_vn   = []
    list_vt   = []
    with open(inpath, "r", errors='replace') as fp:
        line = fp.readline()
        cnt  = 1
        while line:
            # print('cnt: ', cnt, line)
            if len(line)<2:
                line = fp.readline()
                cnt +=1
                continue
            xyz  = []
            xyzn= []
            xyzt= []
            face = []
            mesh = {}
            if line[0] == 'g':
                list_group.append(line[2:])
            if line[0:2] == 'v ':
                vcount = 0
                while line[0:2] == 'v ':
                    xyz.append([float(coord) for coord in line[2:].strip().split()])
                    vcount +=1
                    line = fp.readline()
                    cnt  +=1
                vsplit.append(vcount)
                list_xyz.append(xyz)

            if line[0:2] == 'vn':
                ncount = 0
                while line[0:2] == 'vn':
                    xyzn.append([float(coord) for coord in line[3:].strip().split()])
                    vncount +=1
                    line = fp.readline()
                    cnt  +=1
                nsplit.append(ncount)
                list_vn.append(xyzn)

            if line[0:2] == 'vt':
                tcount = 0
                while line[0:2] == 'vt':
                    xyzt.append([float(coord) for coord in line[3:].strip().split()])
                    tcount +=1
                    line = fp.readline()
                    cnt  +=1
                tsplit.append(tcount)
                list_vt.append(xyzt)

            # it has intermediate g/obj
            if line[0] == 'f':
                fcount = 0
                while line[0] == 'f':
                    face.append([num for num in line[2:].strip().split()])
                    fcount +=1
                    line = fp.readline()
                    cnt +=1
                    if not line:
                        break
                fsplit.append(fcount)
                list_face.append(face)
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            cnt +=1
    # print('vsplit', vsplit, '\n', 'fsplit', fsplit)
    # print("list_mesh", list_mesh)
    dict_mesh['v'] = list_xyz
    dict_mesh['f'] = list_face
    dict_mesh['n'] = list_vn
    dict_mesh['t'] = list_vt
    vsplit_total   = sum(vsplit)
    fsplit_total   = sum(fsplit)

    return dict_mesh, list_group, vsplit, fsplit

def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
    and taken from ObMan dataset (https://github.com/hassony2/obman)
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.

    vertices with the same position but different normals or uvs
    are split into multiple vertices.

    colors are discarded.

    parameters
    ----------
    file_obj : file object
                   containing a wavefront file

    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []

    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes

def save_objmesh(name_obj, dict_mesh, prefix=None):
    with open(name_obj,"w+") as fp:
        if prefix is not None:
            for head_str in prefix:
                fp.write(f'{head_str}\n')
        for i in range(len(dict_mesh['v'])):
            xyz  = dict_mesh['v'][i]
            for j in range(len(xyz)):
                fp.write('v {} {} {}\n'.format(xyz[j][0], xyz[j][1], xyz[j][2]))

            if len(dict_mesh['n']) > 0:
                xyzn  = dict_mesh['n'][i]
                for j in range(len(xyz)):
                    fp.write('vn {} {} {}\n'.format(xyzn[j][0], xyzn[j][1], xyzn[j][2]))

            if len(dict_mesh['t']) > 0:
                xyzt  = dict_mesh['t'][i]
                for j in range(len(xyz)):
                    fp.write('vt {} {}\n'.format(xyzt[j][0], xyzt[j][1]))

            face = dict_mesh['f'][i]
            for m in range(len(face)):
                fp.write('f {} {} {}\n'.format(face[m][0], face[m][1], face[m][2]))
            # fprintf(fid, 'vt %f %f\n',(i-1)/(l-1),(j-1)/(h-1));
            # if (normals) fprintf(fid, 'vn %f %f %f\n', nx(i,j),ny(i,j),nz(i,j)); end
            # Iterate vertex data collected in each material
        fp.write('g\n\n')

def save_multiobjmesh(name_obj, dict_mesh):
    with open(name_obj,"w+") as fp:
        for i in range(len(dict_mesh['v'])):
            xyz  = dict_mesh['v'][i]
            face = dict_mesh['f'][i]
            for j in range(len(xyz)):
                fp.write('v {} {} {}\n'.format(xyz[j][0], xyz[j][1], xyz[j][2]))
            for m in range(len(face)):
                fp.write('f {} {} {}\n'.format(face[m][0], face[m][1], face[m][2]))
            # for name, material in obj_model.materials.items():
            #     # Contains the vertex format (string) such as "T2F_N3F_V3F"
            #     # T2F, C3F, N3F and V3F may appear in this string
            #     material.vertex_format
            #     # Contains the vertex list of floats in the format described above
            #     material.vertices
            #     # Material properties
            #     material.diffuse
            #     material.ambient
            #     material.texture
        fp.write('g mesh\n')
        fp.write('g\n\n')

def get_test_group(all_test_h5, unseen_instances, domain='seen', spec_instances=[], category=None):
    seen_test_h5    = []
    unseen_test_h5  = []
    seen_arti_select = list(np.arange(0, 31, 3)) # todo,  15 * 1 * [24 - 83], half
    seen_arti_select = [str(x) for x in seen_arti_select]

    unseen_frame_select =  list(np.arange(0, 30, 5)) # todo, 6 * 31 * 3
    unseen_frame_select =  [str(x) for x in unseen_frame_select]
    for test_h5 in all_test_h5:
        if test_h5[0:4] in spec_instances or test_h5[-2:] !='h5':
            continue
        name_info      = test_h5.split('.')[0].split('_')
        item           = name_info[0]
        art_index      = name_info[1]
        frame_order    = name_info[2]

        if item in unseen_instances and frame_order in unseen_frame_select :
            unseen_test_h5.append(test_h5)
        elif item not in unseen_instances and art_index in seen_arti_select:
            seen_test_h5.append(test_h5)

    if domain == 'seen':
        test_group = seen_test_h5
    else:
        test_group = unseen_test_h5

    return test_group

def get_full_test(all_test_h5, unseen_instances, domain='seen', spec_instances=[], category=None):
    seen_test_h5    = []
    unseen_test_h5  = []
    for test_h5 in all_test_h5:
        if test_h5[0:4] in spec_instances or test_h5[-2:] !='h5':
            continue
        name_info      = test_h5.split('.')[0].split('_')
        item           = name_info[0]
        art_index      = name_info[1]
        frame_order    = name_info[2]

        if item in unseen_instances:
            unseen_test_h5.append(test_h5)
        elif item not in unseen_instances:
            seen_test_h5.append(test_h5)

    if domain == 'seen':
        test_group = seen_test_h5
    else:
        test_group = unseen_test_h5

    return test_group


def get_demo_h5(all_test_h5, spec_instances=[]):
    demo_full_h5    = []
    for test_h5 in all_test_h5:
        if test_h5[0:4] in spec_instances or test_h5[-2:] !='h5':
            continue
        demo_full_h5.append(test_h5)

    return demo_full_h5

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == '__main__':
    pass
