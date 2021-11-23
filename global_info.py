import os
import sys
import glob
import platform
import collections
import numpy as np

def bp():
    import pdb;pdb.set_trace()

class global_info(object):
    def __init__(self):
        self.name      = 'art6d'
        self.model_type= 'so3net'

        second_path = 'please set this!!!'
        project_path= 'please set this!!!'
        if 'dragon' in platform.uname()[1]:
            second_path = '/home/dragonx/Dropbox/neurips21'
            project_path = '/home/dragonx/Dropbox/neurips21/code'

        self.nocs_real_scale_dict = {'bottle': 0.5, 'bowl': 0.25, 'camera': 0.27,
                           'can': 0.2, 'laptop': 0.5, 'mug': 0.21}

        self.platform_name = platform.uname()[1]
        self.render_path = second_path + '/data/render'
        self.viz_path  = second_path + '/data/images'
        self.grasps_meta = second_path + '/data/grasps'

        self.whole_obj = second_path + '/data/objs'
        self.second_path = second_path
        self.project_path= project_path
        self.categories_list = ['02876657', '03797390', '02880940', '02946921', '03593526', '03624134', '02992529', '02942699', '04074963']
        self.categories = { 'bottle': '02876657',   # 498
                            'mug': '03797390',      # 214
                            'bowl': '02880940',     # 186
                            'can': '02946921',      # 108
                            'jar':  '03593526',     # 596
                            'knife': '03624134',    # 424
                            'cellphone': '02992529',# 831
                            'camera': '02942699',   # 113,
                            'remote': '04074963',   # 66
                            'laptop': '03642806',   # 03642806?
                            'airplane': 'airplane',
                            'chair': 'chair',
                            'car': 'car',
                            'laptop': '03642806',
                            }
        self.categories_id = { '02876657': 'bottle', # 498
                            '03797390': 'mug',  # 214
                            '02880940': 'bowl', # 186
                            '02946921': 'can' , # 108
                            '03593526': 'jar'  ,  # 596
                            '03624134': 'knife' , # 424
                            '02992529': 'cellphone' ,# 831
                            '02942699': 'camera', # 113
                            '04074963': 'remote', # 66
                            '03642806': 'laptop',
                            'airplane': 'airplane',
                            'chair': 'chair',
                            'car': 'car'
                            }
        self.symmetry_dict = np.load(f'{self.project_path}/equi-pose/utils/cache/symmetry.npy', allow_pickle=True).item()
        sym_type = {}
        sym_type['bottle'] = {'y': 36} # use up axis
        sym_type['bowl']   = {'y': 36} # up axis!!!
        sym_type['can']    = {'y': 36, 'x': 2, 'z': 2} # up axis could be also 180 upside down
        sym_type['jar']    = {'y': 36, 'x': 2} # up axis only, + another axis? or 2 * 6D representations
        sym_type['mug']    = {'y': 1}  # up axis + ;
        sym_type['knife']  = {'y': 2, 'x': 2}  # up axis + ;
        sym_type['camera'] = {'y': 1}  # no symmetry; 6D predictions? or addtional axis!!!
        sym_type['remote'] = {'y': 2, 'x': 2}  # symmetric setting, 180 apply to R,
        sym_type['cellphone'] = {'y': 2, 'x': 2} # up axis has 2 groups, x axis has
        sym_type['airplane']= {'y': 1}
        sym_type['chair']= {'y': 1}
        sym_type['car']= {'y': 1}
        sym_type['sofa']= {'y': 1}
        sym_type['laptop']= {'y': 1}
        self.sym_type = sym_type

        delta_R = {}
        delta_T = {}

        # find all pre-computed delta_R, delta_T
        rt_files = glob.glob(f'{project_path}/equi-pose/utils/cache/*.npy')
        for rt_file in rt_files:
            name = rt_file.split('/')[-1].split('.npy')[0]
            if 'nocs_real' in name or 'nocs_real' in name:
                all = name.split('_')
                exp_num, name_dset, category = '_'.join(all[:-3]), '_'.join(all[-3:-1]), all[-1]
            elif 'symmetry' in name:
                continue
            else:
                attrs = rt_file.split('/')[-1].split('.npy')[0].split('_')
                if len(attrs) == 3:
                    exp_num, name_dset, category = attrs
                else:
                    exp_num_part1, exp_num_part2, name_dset, category = attrs
                    exp_num = f'{exp_num_part1}_{exp_num_part2}'
            rt_dict = np.load(rt_file, allow_pickle=True).item()
            delta_R[f'{exp_num}_{name_dset}_{category}'] = rt_dict['delta_r']
            if 'delta_t' in rt_dict:
                delta_T[f'{exp_num}_{name_dset}_{category}'] = rt_dict['delta_t'].reshape(1, 3)
        self.delta_R = delta_R
        self.delta_T = delta_T

if __name__ == '__main__':
    infos = global_info()
