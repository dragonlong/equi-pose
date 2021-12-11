from time import time
import hydra
from omegaconf import OmegaConf
import wandb
import torch

from collections import OrderedDict
from tqdm import tqdm

from datasets.dataset_parser import DatasetParser
from models import get_agent
#
from common.debugger import *
from common.train_utils import cycle
from common.eval_utils import metric_containers
from common.ransac import ransac_delta_pose
from vgtk.functional import so3_mean

from global_info import global_info

@hydra.main(config_path="configs", config_name="pose")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(hydra.utils.get_original_cwd())
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    #
    infos           = global_info()
    my_dir          = infos.second_path
    project_path    = infos.project_path
    categories_id   = infos.categories_id
    categories      = infos.categories

    whole_obj = infos.whole_obj
    sym_type  = infos.sym_type
    cfg.log_dir     = infos.second_path + cfg.log_dir
    cfg.model_dir   = cfg.log_dir + '/checkpoints'
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        os.makedirs(cfg.log_dir + '/checkpoints'
        )
    #>>>>>>>>>>>>>>>>>>>>>> create network and training agent
    if 'airplane' in cfg.category:
        cfg.r_method_type=-1 # one variant
    tr_agent = get_agent(cfg)
    if cfg.use_wandb:
        run_name = f'{cfg.exp_num}_{cfg.category}'
        wandb.init(project="equi-pose", name=run_name)
        wandb.init(config=cfg)
        wandb.watch(tr_agent.net)
    #
    # load checkpoints
    if cfg.use_pretrain or cfg.eval:
        if cfg.ckpt:
            tr_agent.load_ckpt('best', model_dir=cfg.ckpt)
        else:
            tr_agent.load_ckpt('latest')

    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg))
        print(cfg.log_dir)

    #>>>>>>>>>>>>>>>>>>>> dataset <<<<<<<<<<<<<<<<<<<<#
    parser = DatasetParser(cfg)
    train_loader = parser.trainloader
    val_loader   = parser.validloader
    test_loader  = parser.validloader

    if cfg.eval:
        if cfg.pre_compute_delta:
            save_dict = ransac_delta_pose(cfg, train_loader)
            cfg.pre_compute_delta = False

        all_rts, file_name, mean_err, r_raw_err, t_raw_err, s_raw_err = metric_containers(cfg.exp_num, cfg)
        infos_dict = {'basename': [], 'in': [], 'r_raw': [], 'r_gt': [], 't_gt': [], 's_gt': [], 'r_pred': [], 't_pred': [], 's_pred': []}
        track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [], '5deg': [], '5cm': [], '5deg5cm': [], 'chamferL1': [], 'r_acc': [], 'chirality': []}
        num_iteration = 1
        if 'partial' not in cfg.task:
            num_iteration = 5
        for iteration in range(num_iteration):
            cfg.iteration = iteration
            for num, data in enumerate(test_loader):
                BS = data['points'].shape[0]
                idx = data['idx']
                torch.cuda.empty_cache()
                tr_agent.eval_func(data)

                pose_diff = tr_agent.pose_err
                if pose_diff is not None:
                    for key in ['rdiff', 'tdiff', 'sdiff']:
                        track_dict[key] += pose_diff[key].float().cpu().numpy().tolist()
                    print(pose_diff['rdiff'])
                    deg = pose_diff['rdiff'] <= 5.0
                    if cfg.name_dset == 'nocs_real':
                        print('we use real world t metric!!!')
                        cm = pose_diff['tdiff'] <= 0.05/infos.nocs_real_scale_dict[cfg.category]
                    else:
                        cm = pose_diff['tdiff'] <= 0.05
                    degcm = deg & cm
                    for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
                        track_dict[key] += value.float().cpu().numpy().tolist()

                if tr_agent.pose_info is not None:
                    for key, value in tr_agent.pose_info.items():
                        infos_dict[key] += value.float().cpu().numpy().tolist()
                    if 'xyz' in data:
                        input_pts  = data['xyz']
                    else:
                        input_pts  = data['G'].ndata['x'].view(BS, -1, 3).contiguous() # B, N, 3
                    for m in range(BS):
                        basename   = f'{cfg.iteration}_' + data['id'][m] + f'_' + data['class'][m]
                        infos_dict['basename'].append(basename)
                        infos_dict['in'].append(input_pts[m].cpu().numpy())

                if 'completion' in cfg.task:
                    track_dict['chamferL1'].append(torch.sqrt(tr_agent.recon_loss).cpu().numpy().tolist())

                tr_agent.visualize_batch(data, "test")

        print(f'# >>>>>>>> Exp: {cfg.exp_num} for {cfg.category} <<<<<<<<<<<<<<<<<<')
        for key, value in track_dict.items():
            if len(value) < 1:
                continue
            print(key, np.array(value).mean())
            if key == 'rdiff':
                print(np.median(np.array(value)))
            if key == 'tdiff':
                print(np.median(np.array(value)))
        if cfg.save:
            print('--saving to ', file_name)
            np.save(file_name, arr={'info': infos_dict, 'err': track_dict})
        return

    # >>>>>>>>>>>>>>>>>>>>>>>  main training <<<<<<<<<<<<<<<<<<<<< #
    clock = tr_agent.clock #
    val_loader  = cycle(val_loader)
    best_5deg   = 0
    best_chamferL1 = 100
    for e in range(clock.epoch, cfg.nr_epochs):
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            torch.cuda.empty_cache()
            tr_agent.train_func(data)
            # visualize
            if cfg.vis and clock.step % cfg.vis_frequency == 0:
                tr_agent.visualize_batch(data, "train")

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            infos = tr_agent.collect_loss()
            if 'r_acc' in tr_agent.infos:
                infos['r_acc'] = tr_agent.infos['r_acc']
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in infos.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                torch.cuda.empty_cache()
                data = next(val_loader)
                tr_agent.val_func(data)

                if cfg.vis and clock.step % cfg.vis_frequency == 0:
                    tr_agent.visualize_batch(data, "validation")

            if clock.step % cfg.eval_frequency == 0:
                track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [],
                              '5deg': [], '5cm': [], '5deg5cm': [], 'chamferL1': [], 'r_acc': [],
                              'class_acc': []}

                for num, test_data in enumerate(test_loader):
                    if num > 100:
                        break
                    tr_agent.eval_func(test_data)
                    pose_diff = tr_agent.pose_err
                    if pose_diff is not None:
                        for key in ['rdiff', 'tdiff', 'sdiff']:
                            track_dict[key].append(pose_diff[key].cpu().numpy().mean())
                        pose_diff['rdiff'][pose_diff['rdiff']>170] = 180 - pose_diff['rdiff'][pose_diff['rdiff']>170]
                        deg = pose_diff['rdiff'] <= 5.0
                        cm = pose_diff['tdiff'] <= 0.05
                        degcm = deg & cm
                        for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
                            track_dict[key].append(value.float().cpu().numpy().mean())
                    if 'so3' in cfg.encoder_type:
                        test_infos = tr_agent.infos
                        if 'r_acc' in test_infos:
                            track_dict['r_acc'].append(test_infos['r_acc'].float().cpu().numpy().mean())
                    if 'completion' in cfg.task:
                        track_dict['chamferL1'].append(tr_agent.recon_loss.cpu().numpy().mean())

                if cfg.use_wandb:
                    for key, value in track_dict.items():
                        if len(value) < 1:
                            continue
                        wandb.log({f'test/{key}': np.array(value).mean(), 'step': clock.step})
                if np.array(track_dict['5deg']).mean() > best_5deg:
                    tr_agent.save_ckpt('best')
                    best_5deg = np.array(track_dict['5deg']).mean()

                if np.array(track_dict['chamferL1']).mean() < best_chamferL1:
                    tr_agent.save_ckpt('best_recon')
                    best_chamferL1 = np.array(track_dict['chamferL1']).mean()

            clock.tick()
            if clock.step % cfg.save_step_frequency == 0:
                tr_agent.save_ckpt('latest')

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
