import torch
import numpy as np
import sys

EPS = 1e-6
def bp():
    import pdb;pdb.set_trace()

def mean_angular_error(pred_R, gt_R):
    R_diff = torch.matmul(pred_R, gt_R.transpose(1,2).float())
    angles = angle_from_R(R_diff)
    return angles#.mean()
def angle_from_R(R):
    return acos_safe(0.5 * (torch.einsum('bii->b',R) - 1))
    
def rotate_pts_batch(source, target):  # src, tgt [B, P, H, N, 3]
    M = torch.matmul(target.transpose(-1, -2), source)  # [..., 3, N] * [..., N, 3] = [..., 3, 3]
    _M = M.cpu()
    del M
    try:
        U, D, V = torch.svd(_M)  # [B, ..
    except RuntimeError:
        print("torch.svd failed to converge on ", M)
        print("source:", source)
        print("target:", target)
        sys.exit(1)

    '''reflection!'''
    def det(A):  # A: [P, H, 3, 3]
        return torch.sum(torch.cross(A[..., 0, :], A[..., 1, :], dim=-1) * A[..., 2, :], dim=-1)

    device = source.device
    _U, _V = U.to(device), V.to(device)
    del U, V
    U, V = _U, _V
    d = det(torch.matmul(U, V.transpose(-1, -2)))
    mid = torch.zeros_like(U)
    mid[..., 0, 0] = 1.
    mid[..., 1, 1] = 1.
    mid[..., 2, 2] = d

    R = torch.matmul(torch.matmul(U, mid), V.transpose(-1, -2))

    return R


def scale_pts_batch(source, target):  # [B, P, H, N, 3]
    scale = (torch.sum(source * target, dim=(-1, -2)) /
             (torch.sum(source * source, dim=(-1, -2)) + EPS))
    return scale


def translate_pts_batch(source, target):   # [B, P, H, 3, N]
    return torch.mean((target - source), dim=-1, keepdim=True)  # [B, P, H, 3, 1]


def transform_pts_batch(source, target):
    """
    src, tgt: [B, P, H, N, 3]
    """
    source_centered = source - torch.mean(source, -2, keepdim=True)
    target_centered = target - torch.mean(target, -2, keepdim=True)

    rotation = rotate_pts_batch(source_centered, target_centered)
    if rotation is None:
        return None, None, None
    scale = scale_pts_batch(torch.matmul(source_centered,  # [B, P, H, N, 3]
                                         rotation.transpose(-1, -2)),  # [B, P, H, 3, 3]
                            target_centered)

    translation = translate_pts_batch(scale.reshape(scale.shape + (1, 1)) *
                                      torch.matmul(rotation, source.transpose(-1, -2)),
                                      target.transpose(-1, -2))

    return rotation, scale, translation  # [B, P, H, 3, 3], [B, P, H, 1], [B, P, H, 3, 1]


def rotate_pts_mask(source, target, w):  # src, tgt [B, P, 1, N, 3], w [B, P, H, N, 1]
    w = torch.sqrt(w + EPS)
    source = source * w  # already centered
    target = target * w
    return rotate_pts_batch(source, target)


def scale_pts_mask(source, target, w):  # [B, P, 1, N, 3], [B, P, H, N, 1]
    scale = (torch.sum(source * target * w, dim=(-1, -2)) /
             (torch.sum(source * source * w, dim=(-1, -2)) + EPS))
    return scale


def translate_pts_mask(source, target, w):  # [Bs, 3, N], [Bs, N, 1]
    w_shape = list(w.shape)
    w_shape[-2], w_shape[-1] = w_shape[-1], w_shape[-2]
    w = w.reshape(w_shape)  # [Bs, 1, N]
    w_sum = torch.clamp(torch.sum(w, dim=-1, keepdim=True), min=1.0)
    w_normalized = w / w_sum
    return torch.sum((target - source) * w_normalized, dim=-1, keepdim=True)  # [Bs, 3, 1]


def transform_pts_mask(source, target, weights):
    """
    src, tgt [B, N, 3], weights [B, N, 1]
    rotation [B, 3, 3]
    """
    source_center = torch.mean(source, dim=-2, keepdim=True)
    target_center = torch.mean(target, dim=-2, keepdim=True)
    source_centered = (source - source_center)
    target_centered = (target - target_center)

    rotation = rotate_pts_mask(source_centered, target_centered, weights)
    if rotation is None:
        return None, None, None
    scale = scale_pts_mask(torch.matmul(source_centered,  # [B, P, 1, N, 3]
                                        rotation.transpose(-1, -2)),  # [B, P, H, 3, 3]
                           target_centered, weights)
    translation = translate_pts_mask(
        scale.reshape(scale.shape + (1, 1)) * torch.matmul(rotation, source.transpose(-1, -2)),
        target.transpose(-1, -2),
        weights)

    return rotation, scale, translation


def random_choice_noreplace(l, n_sample, num_draw):
    '''
    l: 1-D array or list -> to choose from, e.g. range(N)
    n_sample: sample size for each draw
    num_draw: number of draws

    Intuition: Randomly generate numbers,
    get the index of the smallest n_sample number for each row.
    '''
    l = np.array(l)
    return l[np.argpartition(np.random.rand(num_draw, len(l)),
                             n_sample - 1,
                             axis=-1)[:, :n_sample]]


def pose_fit(source, target, cfg, hard=True):  # [B, N, 3]
    num_hyps = cfg['num_hyps']
    with torch.no_grad():
        B, N = source.shape[:2]
        sample_idx = random_choice_noreplace(torch.tensor(np.arange(N)), 3, B * num_hyps)
        sample_idx = sample_idx.reshape(B, num_hyps, 3)

        src_sampled = source[np.array(range(B)).reshape(-1, 1, 1), sample_idx]
        tgt_sampled = target[np.array(range(B)).reshape(-1, 1, 1), sample_idx]

        rotation, scale, translation = transform_pts_batch(src_sampled, tgt_sampled)
        # [B, H]

        '''score hyp'''

        err = (target.reshape(B, 1, -1, 3, 1)  # [B, 1, N, 3, 1] --> [B, H, N, 3, 1]
               - scale.reshape(scale.shape + (1, 1, 1))  # [B, H] -> [B, H, 1, 1, 1]
               * torch.matmul(rotation.unsqueeze(-3), source.reshape(B, 1, -1, 3, 1))  # [B, H, 1, 3, 3], [B, 1, N, 3, 1]
               - translation.unsqueeze(-3))  # [B, H, 1, 3, 1]
        err = torch.norm(err.reshape(err.shape[:-2] + (3,)), p=2, dim=-1)  # [B, H, N, 3, 1]  -> [B, H, N]

        if hard:
            inliers = (err < cfg['inlier_th']) * 1.0
        else:
            inliers = 1.0 - torch.sigmoid(cfg['inlier_beta'] * (err - cfg['inlier_th']))  # [B, H, N]

        # inliers: [B, H, N], part_mask [B, P, N] --> [B, 1, H, N] * [B, P, 1, N] = [B, P, H, N]
        score = torch.sum(inliers, dim=-1)  # [B, H, N] -> [B, H]

        '''record best model (w.r.t. #inliers)'''
        best_idx = torch.argmax(score, dim=-1)  # [B, H] -> [B]
        best_idxx = (torch.tensor(range(B)).to(rotation.device), best_idx)

        '''refine hyp'''
        if cfg['refine']:
            chosen_inliers = inliers[best_idxx]  # [B, N]
            rotation, scale, translation = transform_pts_mask(source,  # [B, N, 3]
                                                              target,
                                                              chosen_inliers.unsqueeze(-1))

        model = {'rotation': rotation, 'scale': scale, 'translation': translation}  # [B, P]

        return model


from common.d3_utils import rotate_about_axis


def rot_diff_rad(rot1, rot2, chosen_axis=None, flip_axis=False):
    if chosen_axis is not None:
        axis = {'x': 0, 'y': 1, 'z': 2}[chosen_axis]
        y1, y2 = rot1[..., axis], rot2[..., axis]  # [Bs, 3]
        diff = torch.sum(y1 * y2, dim=-1)  # [Bs]
        diff = torch.clamp(diff, min=-1.0, max=1.0)
        rad = torch.acos(diff)
        if not flip_axis:
            return rad
        else:
            return torch.min(rad, np.pi - rad)

    else:
        mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
        diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
        diff = (diff - 1) / 2.0
        diff = torch.clamp(diff, min=-1.0, max=1.0)
        return torch.acos(diff)


def rot_diff_degree(rot1, rot2, chosen_axis=None, flip_axis=False):
    return rot_diff_rad(rot1, rot2, chosen_axis=chosen_axis, flip_axis=flip_axis) / np.pi * 180.0


infos = {}

def compute_pose_diff(nocs_gt, nocs_pred, target, category):
    gt_cfg = {'num_hyps': 16, 'inlier_th': 0.05, 'refine': True}
    pred_cfg = {'num_hyps': 64, 'inlier_th': 0.05, 'refine': True}

    gt_model = pose_fit(nocs_gt, target, gt_cfg)
    pred_model = pose_fit(nocs_pred, target, pred_cfg)

    sym_dict = infos.sym_type[category]
    all_rmats = [np.eye(3)]
    chosen_axis = None
    for key, M in sym_dict.items():
        if M > 20:
            chosen_axis = key
        next_rmats = []
        for k in range(M):
            rmat = rotate_about_axis(2 * np.pi * k / M, axis=key)
            for old_rmat in all_rmats:
                next_rmats.append(np.matmul(rmat, old_rmat))
        all_rmats = next_rmats

    rmats = torch.from_numpy(np.array(all_rmats).astype(np.float32)).to(nocs_pred.device)  # [M, 3, 3]

    gt_rot = gt_model['rotation']  # [B, 3, 3]
    pred_rot = pred_model['rotation']
    gt_rots = torch.matmul(gt_rot.unsqueeze(1), rmats.unsqueeze(0))  # [B, M, 3, 3]
    rot_err = rot_diff_degree(gt_rots, pred_rot.unsqueeze(1), chosen_axis=chosen_axis)  # [B, M]
    rot_err, rot_idx = torch.min(rot_err, dim=-1)  # [B], [B]
    gt_rot = gt_rots[torch.tensor(np.arange(len(gt_rots))).to(rot_idx.device), rot_idx]  # [B, 3, 3]

    scale_err = torch.abs(gt_model['scale'] - pred_model['scale'])

    def transform_center(rot, trans, scale):
        center = torch.ones((1, 3, 1)).to(rot.device) * 0.5
        return scale.unsqueeze(-1).unsqueeze(-1) * torch.matmul(rot, center) + trans
    gt_trans   = transform_center(gt_rot, gt_model['translation'], gt_model['scale'])
    pred_trans = transform_center(pred_rot, pred_model['translation'], pred_model['scale'])
    trans_err   = gt_trans - pred_trans
    trans_err = torch.sqrt((trans_err ** 2).sum((-1, -2)))

    """
    trans_err = gt_model['translation'] - pred_model['translation']  # [B, 3, 1]
    trans_err = torch.sqrt((trans_err ** 2).sum(-1, -2))
    """
    pose_err = {'rdiff': rot_err, 'tdiff': trans_err, 'sdiff': scale_err}
    pose_info= {'r_gt': gt_rot, 't_gt': gt_trans, 's_gt': gt_model['scale'], 'r_pred': pred_rot, 't_pred': pred_trans, 's_pred': gt_model['scale']}
    return pose_err, pose_info
