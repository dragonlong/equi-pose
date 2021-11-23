# import numpy as np
import os
import sys
import time
import json
import h5py
import pickle
import numpy as np
import argparse
import platform
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pytransform3d.rotations import *
import torch

import __init__
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object, hist_show, plot_distribution
from common.data_utils import get_demo_h5, get_full_test, save_objmesh, fast_load_obj, get_obj_mesh, load_pickle
from common.debugger import *

from common.d3_utils import transform_pcloud
from global_info import global_info

infos     = global_info()
my_dir    = infos.second_path
second_path = infos.second_path
render_path = infos.render_path
viz_path  = infos.viz_path
grasps_meta  = infos.grasps_meta
whole_obj = infos.whole_obj


categories_id = infos.categories_id
project_path = infos.project_path
#>>>>>>>>>>>>>>>>>>>>>>>>>>> misc <<<<<<<<<<<<<<<<<<<<<<<<<<<< #
def fetch_data_entry(main_exp, args):
    if args.item == 'obman':
        second_path = second_path + '/model'
        test_h5_path    = second_path + f'/{args.item}/{args.exp_num}/preds/{args.domain}'
        test_group = os.listdir(test_h5_path)
    else:
        if args.domain == 'demo':
            second_path       = second_path + '/results/demo'
            test_h5_path    = second_path + '/{}'.format(main_exp)
            test_group      = get_demo_h5(os.listdir(test_h5_path))
        else:
            second_path       = second_path + '/model' # testing
            print('---checking results from ', second_path)
            test_h5_path    = second_path + f'/{args.item}/{args.exp_num}/preds/{args.domain}'
            test_group      = get_full_test(os.listdir(test_h5_path), unseen_instances, domain=args.domain, spec_instances=special_ins)
    return second_path, test_h5_path, test_group

def fetch_exp_nums(args):
    if args.item == 'obman':
        exp_nums = [args.exp_num]
    else:
        exp_nums = ['0.94', args.exp_num, '1.1']
        # 0.94: # retrain with larger learning rate, 0.1-0.2, add center + confidence prediction after 1st epoch
        # # regressionR(6D), regressionT, partcls loss, NOCS loss, hand vertices, joints loss, using L1 loss
    return exp_nums

def split_basename(basename, args):
    if args.item == 'obman':
        try:
            frame_order, instance, art_index, grasp_ind  = basename.split('_')[0:3] + ['0']
        except:
            instance, art_index, grasp_ind, frame_order = '0', '0', '0', basename
    else:
        instance, art_index, grasp_ind, frame_order = basename.split('_')[0:4]

    return instance, art_index, grasp_ind, frame_order

def get_parts_ind(hf, num_parts=4, gt_key='partcls_per_point_gt', pred_key='partcls_per_point_pred', verbose=False):
    mask_gt        =  hf[gt_key][()]
    mask_pred      =  hf[pred_key][()]
    part_idx_list_gt     = []
    part_idx_list_pred   = []

    if len(mask_pred.shape) > 1:
        mask_pred = mask_pred.transpose(1, 0)
        cls_per_pt_pred      = np.argmax(mask_pred, axis=1)
    else:
        cls_per_pt_pred = mask_pred

    for j in range(num_parts):
        part_idx_list_gt.append(np.where(mask_gt==j)[0])
        part_idx_list_pred.append(np.where(cls_per_pt_pred==j)[0])

    return part_idx_list_gt, part_idx_list_pred

def get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False):
    input_pts = hf['P_gt'][()]
    if input_pts.shape[1] > 10:
        input_pts = input_pts.transpose(1, 0)
    gt_parts  = [input_pts[part_idx_list_gt[j], :3] for j in range(num_parts)]
    pred_parts= [input_pts[part_idx_list_gt[j], :3] for j in range(num_parts)]
    input_pts = input_pts[:, :3]
    if verbose:
        plot3d_pts([[input_pts]], [['pts']], s=2**2, title_name=['inputs'], sub_name=str(i), axis_off=False, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([gt_parts], [['Part {}'.format(j) for j in range(num_parts)]], s=2**2, title_name=['GT seg'], sub_name=str(i),  axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        # plot3d_pts([gt_parts[:3]], [['Part {}'.format(j) for j in range(3)]], s=2**2, title_name=['GT object'], sub_name=str(i),  axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([pred_parts], [['Part {}'.format(j) for j in range(num_parts)]], s=2**2, title_name=['Pred seg'], sub_name=str(i), axis_off=False, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)

    return input_pts, gt_parts, pred_parts

def get_parts_nocs(hf, part_idx_list_gt, part_idx_list_pred, nocs='part', num_parts=4, verbose=False):
    nocs_err   = []
    scale_list = []
    rt_list    = []
    nocs_gt        =  {}
    nocs_pred      =  {}
    nocs_gt['pn']  =  hf['nocs_per_point_gt'][()].transpose(1, 0)
    nocs_pred['pn']=  hf['nocs_per_point_pred'][()].transpose(1, 0)
    for j in range(0, num_parts):
    # for j in range(num_parts-1):
        if nocs == 'part':
            a = nocs_gt['pn'][part_idx_list_gt[j], :]
            b1 = nocs_pred['pn'][part_idx_list_pred[j], 3*j:3*(j+1)]
            b = nocs_pred['pn'][part_idx_list_gt[j], 3*j:3*(j+1)]
        else:
            a = nocs_gt['gn'][part_idx_list_gt[j], :]
            if nocs_pred['gn'].shape[1] ==3:
                b = nocs_pred['gn'][part_idx_list_gt[j], :3]
            else:
                b = nocs_pred['gn'][part_idx_list_gt[j], 3*j:3*(j+1)]

        c = input_pts[part_idx_list_gt[j], :] # point cloud
        c1 = input_pts[part_idx_list_pred[j], :] # pred
        nocs_err.append(np.mean(np.linalg.norm(a - b, axis=1)))
        print('')
        if verbose and j!=0:
            print(f'Part {j} mean nocs error: {nocs_err[-1]}')
            print(f'Part {j} GT nocs: \n', a)
            print(f'Part {j} Pred nocs: \n', b)
            # plot3d_pts([[a],[b]], [['part {}'.format(j)], ['part {}'.format(j)]], s=5, title_name=['GT', 'Pred'], limits = [[0, 1], [0, 1], [0, 1]])
            plot3d_pts([[a], [a]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5, title_name=['GT NPCS', 'Pred NPCS'], color_channel=[[a], [b]], save_fig=True, limits = [[0, 1], [0, 1], [0, 1]], sub_name='{}'.format(j))
            plot3d_pts([[a]], [['Part {}'.format(j)]], s=5, title_name=['NOCS Error'], color_channel=[[np.linalg.norm(a - b, axis=1)]], limits = [[0, 1], [0, 1], [0, 1]], colorbar=True)
            plot3d_pts([[a], [b]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5, title_name=['GT NPCS', 'Pred NPCS'], color_channel=[[a], [a]], save_fig=True, limits = [[0, 1], [0, 1], [0, 1]], sub_name='{}'.format(j))
            # plot3d_pts([[a], [c]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5, title_name=['GT NOCS', 'point cloud'],limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]], color_channel=[[a], [a]], save_fig=True, sub_name='{}'.format(j))
        # s, r, t, rt = estimateSimilarityUmeyama(a.transpose(), c.transpose())
        # rt_list.append(compose_rt(r, t))
        # scale_list.append(s)
        # s, r, t, rt = estimateSimilarityUmeyama(b1.transpose(), c1.transpose())
        # rt_list.append(compose_rt(r, t))
        # scale_list.append(s)
    # if args.viz and j==3:
    #     d = np.matmul(a*s, r) + t
    #     plot3d_pts([[d, c]], [['Part {} align'.format(j), 'Part {} cloud'.format(j)]], s=15, title_name=['estimation viz'], color_channel=[[a, a]], save_fig=True, sub_name='{}'.format(j))
    return nocs_gt, nocs_pred

def plot_hand_skeleton(hand_joints):
    pass

def get_pca_stuff(second_path):
    pca_hand = np.load(second_path + f'/data/pickle/hand_pca.npy',allow_pickle=True).item()
    mean_u  = pca_hand['M'] # 63
    eigen_v = pca_hand['E'] # 30*63
    return eigen_v, mean_u

def add_contacts_evaluation(stat_dict, contact_pts, contacts_num):
    stat_dict['gt'].append(contact_pts[0])
    stat_dict['pred'].append(contact_pts[1])
    stat_dict['gt_cnt']+=contacts_num[0]
    stat_dict['pred_cnt']+=contacts_num[1]

def eval_contacts(domain, contacts_stat, save_path):
    contacts_err = np.linalg.norm(np.concatenate(contacts_stat['gt'], axis=0) - np.concatenate(contacts_stat['pred'], axis=0), axis=1)
    mean_contacts_err = np.mean(contacts_err)
    contacts_miou    = contacts_stat['pred_cnt']/contacts_stat['gt_cnt']
    np.savetxt(f'{save_path}/contact_err_{domain}.txt', contacts_err, fmt='%1.5f', delimiter='\n')

    print(f'>>>>>>>>>>>>>   {domain}    <<<<<<<<<<<< \n contacts_stat:\n contacts_err: {contacts_err}\n mean_contacts_err: {mean_contacts_err}\n contacts_miou:{contacts_miou}\n')

def save_pose_pred(all_rts, filename):
    print('saving to ', file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(all_rts, f)

    for j in range(num_parts):
        print('mean rotation err of part {}: \n'.format(j), 'baseline: {}'.format(np.array(r_raw_err['pred'][j]).mean()))
        print('mean translation err of part {}: \n'.format(j), 'baseline: {}'.format(np.array(t_raw_err['pred'][j]).mean()))

def metric_containers(save_exp, args, num_parts=2, extra_key=None):
    file_name       = my_dir + '/results/test_pred/{}/{}_{}_{}_rt_pn_general.npy'.format(args.item, save_exp, args.domain, args.nocs)
    save_path       = my_dir + '/results/test_pred/{}/'.format(args.item)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_rts   = {}
    mean_err  = {'pred': [], 'nonlinear': []}
    if num_parts   == 1:
        r_raw_err   = {'pred': [[]], 'nonlinear': [[]]}
        t_raw_err   = {'pred': [[]], 'nonlinear': [[]]}
        s_raw_err   = {'pred': [[]], 'nonlinear': [[]]}
    elif num_parts   == 2:
        r_raw_err   = {'pred': [[], []], 'nonlinear': [[], []]}
        t_raw_err   = {'pred': [[], []], 'nonlinear': [[], []]}
        s_raw_err   = {'pred': [[], []], 'nonlinear': [[], []]}
    elif num_parts == 3:
        r_raw_err   = {'pred': [[], [], []], 'nonlinear': [[], [], []]}
        t_raw_err   = {'pred': [[], [], []], 'nonlinear': [[], [], []]}
        s_raw_err   = {'pred': [[], [], []], 'nonlinear': [[], [], []]}
    elif num_parts == 4:
        r_raw_err   = {'pred': [[], [], [], []], 'nonlinear': [[], [], [], []]}
        t_raw_err   = {'pred': [[], [], [], []], 'nonlinear': [[], [], [], []]}
        s_raw_err   = {'pred': [[], [], [], []], 'nonlinear': [[], [], [], []]}
    return all_rts, file_name, mean_err, r_raw_err, t_raw_err, s_raw_err

def post_summary(all_rts, file_name=None, args=None, r_raw_err=None, t_raw_err=None, extra_key=None):
    if args.save:
        print('--saving to ', file_name)
        np.save(file_name, arr=all_rts)

    # evaluate per category as well
    xyz_err_dict = {}
    rpy_err_dict = {}
    for key, value_dict in all_rts.items():
        category_name = key.split('_')[-1]
        if category_name not in xyz_err_dict:
            xyz_err_dict[category_name] = []
            rpy_err_dict[category_name] = []
        if isinstance(value_dict['rpy_err']['pred'], list):
            rpy_err_dict[category_name].append(value_dict['rpy_err']['pred'][0])
        else:
            rpy_err_dict[category_name].append(value_dict['rpy_err']['pred'])
        if isinstance(value_dict['xyz_err']['pred'], list):
            xyz_err_dict[category_name].append(value_dict['xyz_err']['pred'][0])
        else:
            xyz_err_dict[category_name].append(value_dict['xyz_err']['pred'])

    # in total
    if r_raw_err is not None:
        for j in range(1, num_parts):
            print('mean rotation err of part {}: \n'.format(j), 'baseline: {}'.format(np.array(r_raw_err['pred'][j]).mean()))
            print('mean translation err of part {}: \n'.format(j), 'baseline: {}'.format(np.array(t_raw_err['pred'][j]).mean()))
    all_categorys = xyz_err_dict.keys()
    print('category\trotation error\ttranslation error')
    for category in all_categorys:
        print(f'{category}\t{np.array(rpy_err_dict[category]).mean():0.4f}\t{np.array(xyz_err_dict[category]).mean():0.4f}')

    num_parts = 1
    print('For {} object, {}, 5 degrees accuracy is: '.format(args.domain, args.nocs))
    for category in all_categorys:
        r_err = np.array(rpy_err_dict[category])
        num_valid = r_err.shape[0]
        r_acc = []
        for j in range(num_parts):
            idx = np.where(r_err < 5)[0]
            acc   = len(idx) / num_valid
            r_acc.append(acc)
        print(category, " ".join(["{:0.4f}".format(x) for x in r_acc]))
    print('\n')
    # 5 degrees & 0.05
    print('For {} object, {}, 5 degrees, 5 cms accuracy is: '.format(args.domain, args.nocs))
    for category in all_categorys:
        num_valid = r_err.shape[0]
        rt_acc = []
        t_err  = np.array(xyz_err_dict[category])
        for j in range(num_parts):
            idx = np.where(r_err < 5)[0]
            acc   = len(np.where( t_err[idx] < 0.05 )[0]) / num_valid
            rt_acc.append(acc) # two modes
        print(category, " ".join(["{:0.4f}".format(x) for x in rt_acc]))
    print('\n')

    if extra_key is not None:
        plot_distribution(rpy_err_dict[category_name], labelx='r_err', labely='frequency', title_name=f'rotation_error_{extra_key}', sub_name=args.exp_num, save_fig=True)
        plot_distribution(xyz_err_dict[category_name], labelx='t_err', labely='frequency', title_name=f'translation_error_{extra_key}', sub_name=args.exp_num, save_fig=True)
    else:
        plot_distribution(rpy_err_dict[category_name], labelx='r_err', labely='frequency', title_name='rotation_error', sub_name=args.exp_num, save_fig=True)
        plot_distribution(xyz_err_dict[category_name], labelx='t_err', labely='frequency', title_name='translation_error', sub_name=args.exp_num, save_fig=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--category', default='', help='exact object category')
    parser.add_argument('--domain', default='seen', help='which sub test set to choose')
    parser.add_argument('--exp_num', default='0.8', required=False) # default is 0.3
    parser.add_argument('--nocs', default='part', help='which sub test set to choose')
    #
    parser.add_argument('--hand', action='store_true', help='whether to visualize hand')
    parser.add_argument('--contact', action='store_true', help='whether to visualize hand')
    parser.add_argument('--vote', action='store_true', help='whether to vote hand joints')
    parser.add_argument('--mano', action='store_true', help='whether to use mano')
    parser.add_argument('--save', action='store_true', help='save err to pickles')
    parser.add_argument('--save_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--show_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--viz', action='store_true', help='whether to viz')
    parser.add_argument('--is_special', action='store_true', help='whether a 360 symmetric object')
    parser.add_argument('--verbose', action='store_true', help='whether to viz')

    args = parser.parse_args()

    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    unseen_instances= dset_info.test_list
    special_ins     = dset_info.spec_list
    main_exp        = args.exp_num
    root_dset       = second_path + '/data'
    second_path, test_h5_path, test_group = fetch_data_entry(main_exp, args)

    print('---we have {} testing data for {} {}'.format(len(test_group), args.domain, args.item))

    save_path = second_path + f'/pickle/{args.item}/{args.exp_num}'
    file_name = second_path + f'/pickle/{args.item}/{args.exp_num}/{args.domain}_hand.pkl'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    start_time = time.time()
    error_all = []
    contacts_stat = {'gt': [], 'pred': [], 'gt_cnt': 0, 'pred_cnt': 0}

    exp_nums = fetch_exp_nums(args) # contacts, mano, translation
    all_rts, file_name, mean_err, r_raw_err, t_raw_err, s_raw_err = metric_containers(main_exp, args)
    # if os.path.exists(file_name):
    #     all_rts = load_pickle(file_name)
    #     post_summary(all_rts, file_name, args, r_raw_err=None, t_raw_err=None)

    # # get_val_dataset():
    # cache_folder = os.path.join(project_path, "haoi-pose/utils/cache", "cache", 'obman')
    # cache_path = os.path.join(cache_folder, "{}_{}_mode_{}.pkl".format('train', '1.0', 'all'))
    # with open(cache_path, "rb") as cache_f:
    #     annotations = pickle.load(cache_f)
    # meta_infos = annotations["meta_infos"]
    # for i in range(10):
    for i in range(len(test_group)):
        h5_files   = []
        hf_list    = []
        h5_file    =  test_h5_path + '/' + test_group[i]
        for sub_exp in exp_nums:
            h5_files.append( h5_file.replace(args.exp_num, sub_exp))
        print('')
        print('----------Now checking {}: {}'.format(i, h5_file))
        for h5_file in h5_files:
            hf             =  h5py.File(h5_file, 'r')
            basename       =  hf.attrs['basename']
            check_h5_keys(hf, verbose=args.verbose)
            hf_list.append(hf)

        hf = hf_list[0]
        basename       =  hf.attrs['basename']
        # 1. get pts and gt contacts
        instance, art_index, grasp_ind, frame_order = split_basename(basename, args)
        part_idx_list_gt, part_idx_list_pred = get_parts_ind(hf, num_parts=num_parts)
        print('---get_parts_pcloud')
        input_pts, gt_parts, pred_parts      = get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=num_parts, verbose=True)

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> optional <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
        print('---get_parts_nocs')
        nocs_gt, nocs_pred                   = get_parts_nocs(hf, part_idx_list_gt, part_idx_list_pred, num_parts=num_parts, verbose=True)
        # nocs_noisy = nocs_gt['pn'] + np.random.rand(nocs_gt['pn'].shape[0], 3) * 1.0
        # nocs_noisy = np.concatenate([nocs_noisy, nocs_noisy], axis=1)
        try:
            rts_dict = compute_pose_ransac(nocs_gt['pn'], nocs_pred['pn'], input_pts, part_idx_list_pred, num_parts, basename, r_raw_err, t_raw_err, s_raw_err, \
                    partidx_gt=part_idx_list_gt, category=args.category, is_special=args.is_special, verbose=False)

            all_rts[basename]  = rts_dict
        except:
            continue
        ############################## optional ends here #######################
        if args.contact:
            contact_pts, contacts_num = get_contacts_camera(hf, basename, verbose=args.verbose)
            add_contacts_evaluation(contacts_stat, contact_pts, contacts_num)
            if args.verbose:
                plot3d_pts([[input_pts, contact_pts[0], contact_pts[1]]], [['input', 'GT contact', 'Pred contact']], s=[1**2, 10**2, 10**2], mode='continuous',  limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]], title_name=['input + contact pts'])

            # 2. check gt vote offset
            get_contacts_vote(hf, basename, input_pts, verbose=args.verbose)

        # combined visualization
        if args.viz:
            plt.show()
            plt.close()

    #     hf = hf_list[1]
    #     instance, art_index, grasp_ind, frame_order = basename.split('_')[0:4]
    #     part_idx_list_gt, part_idx_list_pred = get_parts_ind(hf, num_parts=4)
    #     input_pts, gt_parts, pred_parts      = get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False)
    #     nocs_gt, nocs_pred                   = get_parts_nocs(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False)

        if args.hand:
            if args.vote:
                results = vote_hand_joints_cam(hf, input_pts, part_idx_list_gt, verbose=args.verbose)
            if args.mano:
                verts_canon, faces_canon, joints_canon = get_hand_mesh_canonical(hf, verbose=False) # in hand canonical space
            else:
                hand_joints_cam  = get_hand_regression_camera(hf, input_pts, verbose=args.verbose)

            contact_pts = get_contacts_camera_raw(hf, basename, verbose=args.verbose)
            hand_trans  = get_hand_trans2camera(hf, basename, joints_canon, extra_hf=hf_list[2], category=args.item, verbose=False)

            hand_vertices_cam, hand_faces_cam, hand_joints_cam  = get_hand_mesh_camera(verts_canon, faces_canon, joints_canon, hand_trans, verbose=False)

            mean_hand = hf['mean_hand_gt'][()]

            # object poses estimation

            # obj_vertices, obj_faces = get_obj_mesh(basename, verbose=args.verbose)
            # obj_vertices_tf         = transform_pcloud(np.copy(obj_vertices), RT=hf['extrinsic_params_gt'][()], extra_R=None)
            #
            # # cononical
            # refer_path = '/home/dragon/Documents/CVPR2020/dataset/shape2motion/objects/eyeglasses/{}/part_objs/{}.obj'
            # name_list = ['none_motion', 'dof_rootd_Aa002_r', 'dof_rootd_Aa001_r']
            # if args.verbose:
            #     for part_name in name_list:
            #         part_vertices, part_faces = get_obj_mesh(basename, full_path=refer_path.format(instance, part_name), verbose=True)

            # contacts visualization
            if args.verbose:
            #     # hand
                # plot_hand_w_object(obj_verts=hand_vertices_cam[1], obj_faces=hand_faces_cam[1], hand_verts=hand_vertices_cam[0], hand_faces=hand_faces_cam[0], s=5, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
            #     # print_group(['obj_vertices_tf', 'obj_faces', 'hand_vertices_cam[0]', 'hand_faces_cam[0]',  'contact_pts', 'gt_parts'], [obj_vertices_tf.shape, obj_faces.shape, hand_vertices_cam[0].shape, hand_faces_cam[0].shape, contact_pts.shape, gt_parts[-1].shape])
                plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=obj_vertices_tf, hand_faces=obj_faces, s=20, save=False, mode='continuous')
                # plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=obj_vertices_tf, hand_faces=obj_faces, s=20, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
            #     plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=hand_vertices_cam[1], hand_faces=hand_faces_cam[1], s=20, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
            #
            # combined visualization
            if args.viz:
                plt.show()
                plt.close()

            error_dist = hand_joints_cam[1] - hand_joints_cam[0]
            error_all.append(error_dist) # J, 3
    if args.contact:
        eval_contacts(args.domain, contacts_stat, save_path)

    if args.hand:
        error_norm = np.mean( np.linalg.norm(np.stack(error_all, axis=0), axis=2), axis=0)
        print('experiment: ', args.exp_num)
        for j, err in enumerate(error_norm):
            print(err)
    post_summary(all_rts, file_name, args, r_raw_err=None, t_raw_err=None)
    end_time = time.time()
    print(f'---it takes {end_time-start_time} seconds')
