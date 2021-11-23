import torch
import numpy as np
import numpy.testing as npt
# PyTorch-backed implementations


def normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(norm)


def assert_normalized(q, atol=1e-3):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1)
    #norm = norm.cpu()
    norm_check =  (norm - 1.0).abs()
    try:
        assert torch.max(norm_check) < atol
    except:
        print("normalization failure: {}.".format(torch.max(norm_check)))
        return -1
    return 0
    #npt.assert_allclose(norm_check, np.zeros_like(norm_check), rtol=0, atol=atol)

def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1*qb_2 - qa_2*qb_1
    q_mult_1 = qa_2*qb_0 - qa_0*qb_2
    q_mult_2 = qa_0*qb_1 - qa_1*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = torch.empty(3,3)

    rot_mat[0,0] = aa + bb - cc - dd
    rot_mat[0,1] = 2 * (bc + ad)
    rot_mat[0,2] = 2 * (bd - ac)

    rot_mat[1,0] = 2 * (bc - ad)
    rot_mat[1,1] = aa + cc - bb - dd
    rot_mat[1,2] = 2 * (cd + ab)

    rot_mat[2,0] = 2 * (bd + ac)
    rot_mat[2,1] = 2 * (cd - ab)
    rot_mat[2,2] = aa + dd - bb - cc

    return rot_mat

def rotation_matrix_batch(axis, theta, device):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0) # BS * 1
    b = -axis[0] * torch.sin(theta / 2.0)
    c = -axis[1] * torch.sin(theta / 2.0)
    d = -axis[2] * torch.sin(theta / 2.0)

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    rot_mat = torch.empty(aa.shape[0],3,3).cuda()

    rot_mat[:,0,0] = aa + bb - cc - dd
    rot_mat[:,0,1] = 2 * (bc + ad)
    rot_mat[:,0,2] = 2 * (bd - ac)

    rot_mat[:,1,0] = 2 * (bc - ad)
    rot_mat[:,1,1] = aa + cc - bb - dd
    rot_mat[:,1,2] = 2 * (cd + ab)

    rot_mat[:,2,0] = 2 * (bd + ac)
    rot_mat[:,2,1] = 2 * (cd - ab)
    rot_mat[:,2,2] = aa + dd - bb - cc

    return rot_mat

def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    q = torch.unsqueeze(q, 1)*ones_x

    q_conj = torch.cat([ q[:, :, [0]] , -1*q[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, [0]]*0, X ], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]

def multiply(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    assert q.shape == r.shape

    original_shape = q.shape

    real1, im1 = q.split([1, 3], dim=-1)
    real2, im2 = r.split([1, 3], dim=-1)

    real = real1*real2 -  torch.sum(im1*im2, dim=-1, keepdim=True)
    im = real1*im2 + real2*im1 + im1.cross(im2, dim=-1)
    return torch.cat((real, im), dim=-1)


def conjugate(q):
    assert q.shape[-1] == 4
    w, xyz = q.split([1, 3], dim=-1)
    return torch.cat((w, -xyz), dim=-1)


def rotate(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # assert_normalized(q)

    zeros = torch.zeros(original_shape[:-1]+[1]).cuda()
    qv = torch.cat((zeros, v), dim=-1)

    result = multiply(multiply(q, qv), conjugate(q))
    _, xyz = result.split([1, 3], dim=-1)
    return xyz.view(original_shape)


def relative_angle(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    assert q.shape == r.shape

    assert_normalized(q)
    assert_normalized(r)

    dot_product = torch.sum(q*r, dim=-1)
    angle = 2 * torch.acos(torch.clamp(dot_product.abs(), min=-1, max=1))

    return angle


def unit_quaternion_to_matrix(q):
    assert_normalized(q)
    w, x, y, z= torch.unbind(q, dim=-1)
    matrix = torch.stack(( 1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y* w,
                        2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,
                        2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x -2*y*y),
                        dim=-1)
    matrix_shape = list(matrix.shape)[:-1]+[3,3]
    return matrix.view(matrix_shape).contiguous()


def matrix_to_unit_quaternion(matrix):
    assert matrix.shape[-1] == matrix.shape[-2] == 3
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    trace = 1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    trace = torch.clamp(trace, min=0.)
    r = torch.sqrt(trace)
    s = 1.0 / (2 * r + 1e-7)
    w = 0.5 * r
    x = (matrix[..., 2, 1] - matrix[..., 1, 2])*s
    y = (matrix[..., 0, 2] - matrix[..., 2, 0])*s
    z = (matrix[..., 1, 0] - matrix[..., 0, 1])*s

    q = torch.stack((w, x, y, z), dim=-1)

    return normalize(q)


def axis_theta_to_quater(axis, theta):  # axis: [Bs, 3], theta: [Bs]
    w = torch.cos(theta / 2.)  # [Bs]
    u = torch.sin(theta / 2.)  # [Bs]
    xyz = axis * u.unsqueeze(-1)  # [Bs, 3]
    new_q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)  # [Bs, 4]
    new_q = normalize(new_q)
    return new_q


def quater_to_axis_theta(quater):
    quater = normalize(quater)
    cosa = quater[..., 0]
    sina = torch.sqrt(1 - cosa ** 2)
    norm = sina.unsqueeze(-1)
    mask = (norm < 1e-8).float()
    axis = quater[..., 1:] / torch.max(norm, mask)
    theta = 2 * torch.acos(torch.clamp(cosa, min=-1, max=1))
    return axis, theta


def axis_theta_to_matrix(axis, theta):
    quater = axis_theta_to_quater(axis, theta)  # [Bs, 4]
    return unit_quaternion_to_matrix(quater)


def matrix_to_axis_theta(matrix):
    quater = matrix_to_unit_quaternion(matrix)
    return quater_to_axis_theta(quater)


def matrix_to_rotvec(matrix):
    axis, theta = matrix_to_axis_theta(matrix)
    theta = theta % (2 * np.pi) + 2 * np.pi
    return axis * theta.unsqueeze(-1)


def rotvec_to_axis_theta(rotvec):
    theta = torch.norm(rotvec, dim=-1, keepdim=True)  # [Bs, 1]
    mask = (theta < 1e-8).float()
    axis = rotvec / torch.max(theta, mask)  # [Bs, 3]
    theta = theta.squeeze(-1)  # [Bs]
    return axis, theta


def rotvec_to_matrix(rotvec):  # [Bs, 3]
    axis, theta = rotvec_to_axis_theta(rotvec)
    return axis_theta_to_matrix(axis, theta)


def rotvec_to_euler(rotvec):
    """
    http://euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/index.htm
    """
    axis, theta = rotvec_to_axis_theta(rotvec)
    x, y, z = torch.unbind(axis, dim=-1)
    s, c = torch.sin(theta), torch.cos(theta)
    t = 1 - c

    mask_n = ((x * y * t + z * s) > 0.998).float().unsqueeze(-1)
    heading = 2 * torch.atan2(x * torch.sin(theta / 2),
                              torch.cos(theta / 2))
    attitude = torch.ones_like(heading) * np.pi / 2.0
    bank = torch.zeros_like(heading)
    euler_n = torch.stack([heading, attitude, bank], dim=-1)

    mask_s = ((x * y * t + z * s) < -0.998).float().unsqueeze(-1)
    heading = -2 * torch.atan2(x * torch.sin(theta / 2),
                              torch.cos(theta / 2))
    attitude = -torch.ones_like(heading) * np.pi / 2.0
    bank = torch.zeros_like(heading)
    euler_s = torch.stack([heading, attitude, bank], dim=-1)

    heading = torch.atan2(y * s - x * z * t, 1 - (y * y + z * z) * t)
    attitude = torch.asin(x * y * t + z * s)
    bank = torch.atan2(x * s - y * z * t, 1 - (x * x + z * z) * t)
    euler = torch.stack([heading, attitude, bank], dim=-1)
    mask = torch.ones_like(mask_n) - mask_n - mask_s

    euler_final = mask_n * euler_n + mask_s * euler_s + mask * euler

    return euler_final


def euler_to_rotvec(euler):
    """
    http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToAngle/index.htm
    """
    heading, attitude, bank = torch.unbind(euler, dim=-1)
    c1 = torch.cos(heading / 2)
    s1 = torch.sin(heading / 2)
    c2 = torch.cos(attitude / 2)
    s2 = torch.sin(attitude / 2)
    c3 = torch.cos(bank / 2)
    s3 = torch.sin(bank / 2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2 * c3 - s1s2 * s3
    x = c1c2 * s3 + s1s2 * c3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    angle = 2 * torch.acos(w)
    axis = torch.stack([x, y, z], dim=-1)
    norm = torch.norm(axis, dim=-1, keepdim=True)
    mask = (norm < 1e-8).float()
    axis = axis / torch.max(norm, mask)
    u_axis = torch.zeros_like(axis)
    u_axis[..., 0] = 1.0
    axis_final = mask * u_axis + (1 - mask) * axis
    return axis_final * angle.unsqueeze(-1)


def jitter_quaternion(q, theta):  #[Bs, 4], [Bs, 1]
    new_q = generate_random_quaternion(q.shape).to(q.device)
    dot_product = torch.sum(q*new_q, dim=-1, keepdim=True)  #
    shape = (tuple(1 for _ in range(len(dot_product.shape) - 1)) + (4, ))
    q_orthogonal = normalize(new_q - q * dot_product.repeat(*shape))
    # theta = 2arccos(|p.dot(q)|)
    # |p.dot(q)| = cos(theta/2)
    tile_theta = theta.repeat(shape)
    jittered_q = q*torch.cos(tile_theta/2) + q_orthogonal*torch.sin(tile_theta/2)

    return jittered_q


def project_to_axis(q, v):  # [Bs, 4], [Bs, 3]
    a = q[..., 0]  # [Bs]
    b = torch.sum(q[..., 1:] * v, dim=-1)  # [Bs]
    rad = 2 * torch.atan2(b, a)
    new_q = axis_theta_to_quater(v, rad)
    residual = relative_angle(q, new_q)

    return rad, new_q, residual


def rotate_points_with_rotvec(points, rot_vec): # [Bs, 3], [Bs, 3]
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = torch.norm(rot_vec, dim=-1, keepdim=True)  # [Bs, 1, 1]
    mask = (theta < 1e-8).float()
    v = rot_vec / torch.max(theta, mask)  # [Bs, 1, 1]
    dot = torch.sum(points * v, dim=-1, keepdim=True)  # [Bs, N, 1]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    return cos_theta * points + sin_theta * torch.cross(v, points, dim=-1) + dot * (1 - cos_theta) * v


def rot_diff_rad(rot1, rot2):
    mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
    diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
    diff = (diff - 1) / 2.0
    diff = torch.clamp(diff, min=-1.0, max=1.0)
    return torch.acos(diff)


def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180.0


def generate_random_quaternion(quaternion_shape):
    assert quaternion_shape[-1] == 4
    rand_norm = torch.randn(quaternion_shape)
    rand_q = normalize(rand_norm)
    return rand_q


def noisy_rot_matrix(matrix, rad, type='normal'):
    if type == 'normal':
        theta = torch.abs(torch.randn_like(matrix[..., 0, 0])) * rad
    elif type == 'uniform':
        theta = torch.rand_like(matrix[..., 0, 0]) * rad
    quater = matrix_to_unit_quaternion(matrix)
    new_quater = jitter_quaternion(quater, theta.unsqueeze(-1))
    new_mat = unit_quaternion_to_matrix(new_quater)
    return new_mat


def jitter_rot_matrix(matrix, rad):
    quater = matrix_to_unit_quaternion(matrix)
    new_quater = jitter_quaternion(quater, rad.unsqueeze(-1))
    new_mat = unit_quaternion_to_matrix(new_quater)
    return new_mat


def rotate_around_point(points, rotation, pivot):  # [Bs, N, 3], [Bs, 3, 3], [Bs, 3]
    pivot = pivot.unsqueeze(-2)  # [Bs, 1, 3]
    points = torch.matmul(points - pivot, rotation.transpose(-1, -2))  # [Bs, N, 3], [Bs, 3, 3]
    points = points + pivot
    return points


def convert_ax_angle_to_quat(ax, ang):
    """
    Convert Euler angles to quaternion.
    """
    qw = torch.cos(ang/2)
    qx = ax[0] * torch.sin(ang/2)
    qy = ax[1] * torch.sin(ang/2)
    qz = ax[2] * torch.sin(ang/2)
    quat = torch.stack([qw, qx, qy, qz], dim=1) #
    return quat

def ang2quat(angles):
    # convert from angles to quaternion
    axis = torch.eye(3).float().cuda()
    ang = torch.tanh(angles)

    azimuth = math.pi/6 * ang[...,0]
    elev = math.pi/2 * (ang[...,1])
    cyc_rot = math.pi/3 * (ang[...,2])

    q_az = convert_ax_angle_to_quat(axis[1], azimuth)
    q_el = convert_ax_angle_to_quat(axis[0], elev)
    q_cr = convert_ax_angle_to_quat(axis[2], cyc_rot)
    quat = hamilton_product(q_el.unsqueeze(1), q_az.unsqueeze(1))
    quat = hamilton_product(q_cr.unsqueeze(1), quat)
    quat = quat.squeeze(1)
    return quat

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

if __name__ == '__main__':


    """
    batch_size = 4
    max_jittered_radian = 5/180*np.pi
    for i in range(100):
        q = generate_random_quaternion((batch_size, 4))
        theta = torch.rand((batch_size, 1))*max_jittered_radian
        jittered_q = jitter_quaternion(q, theta)
        angle = relative_angle(q, jittered_q)*180/np.pi
        print(angle)

    bla = torch.tensor([[0, np.pi, np.pi]]).repeat(5, 1)
    bla = torch.randn(1, 3).repeat(5, 1)
    bla = torch.randn(5, 3)
    mat = rotvec_to_matrix(bla)

    cla = matrix_to_rotvec(mat)

    nat = rotvec_to_matrix(cla)

    P = torch.randn(5, 3)

    P1 = rotate_points_with_rotvec(P, bla)
    P2 = rotate_points_with_rotvec(P, cla)
    P3 = torch.matmul(P.reshape(5, 1, 3), mat.transpose(-1, -2))

    print(torch.mean(mat - nat))
    print(torch.mean(P1 - P2))
    print(torch.mean(P1 - P3))

    """

    rotvec = torch.rand((5, 3))
    rotvec = torch.zeros((5, 3))

    euler = rotvec_to_euler(rotvec)

    rot = euler_to_rotvec(euler)

    print(((rot - rotvec) ** 2).sum())
