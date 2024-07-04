import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras


def full_scene_scale(batch):
    """
    Recovers the scale of the scene, defined as the distance between the centroid of
    the cameras to the furthest camera.

    Args:
        batch (dict): batch containing the camera parameters for all cameras in the
            sequence.

    Returns:
        float: scale of the scene.
    """
    cameras = PerspectiveCameras(R=batch["R"], T=batch["T"])
    cc = cameras.get_camera_center()
    centroid = torch.mean(cc, dim=0)

    diffs = cc - centroid
    norms = torch.linalg.norm(diffs, dim=1)

    furthest_index = torch.argmax(norms).item()
    scale = norms[furthest_index].item()
    return scale


def get_permutations(num_images):
    permutations = []
    for i in range(0, num_images):
        for j in range(0, num_images):
            if i != j:
                permutations.append((j, i))

    return permutations


def n_to_np_rotations(num_frames, n_rots):
    """ this function takes in a list of n rotation matrices and returns a list of n-choose-2 relative rotations
    
    Args:
        num_frames (int): number of frames.
        n_rots (torch.Tensor): (N, 3, 3).
    """

    R_pred_rel = []
    permutations = get_permutations(num_frames)
    for i, j in permutations:
        R_pred_rel.append(n_rots[i].T @ n_rots[j])
    R_pred_rel = torch.stack(R_pred_rel)

    return R_pred_rel


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation2, rotation1.transpose(0, 2, 1))
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


# A should be GT, B should be predicted
def compute_optimal_alignment(A, B):
    """
    Compute the optimal scale s, rotation R, and translation t that minimizes:
    || A - (s * B @ R + T) || ^ 2

    Reference: Umeyama (TPAMI 91)

    Args:
        A (torch.Tensor): (N, 3).
        B (torch.Tensor): (N, 3).

    Returns:
        A_hat (torch.Tensor): (N, 3). A after optimal alignment.
        s (float): scale.
        R (torch.Tensor): rotation matrix (3, 3).
        t (torch.Tensor): translation (3,).
    """
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    # normally with R @ B, this would be A @ B.T
    H = (B - B_bar).T @ (A - A_bar)
    U, S, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    variance = torch.sum((B - B_bar) ** 2)
    scale = 1 / variance * torch.trace(torch.diag(S) @ S_prime)
    R = U @ S_prime @ Vh
    t = A_bar - scale * B_bar @ R

    A_hat = scale * B @ R + t
    return A_hat, scale, R, t


def compute_camera_center_error(R_pred, T_pred, R_gt, T_gt, gt_scene_scale):
    cameras_gt = PerspectiveCameras(R=R_gt, T=T_gt)
    cc_gt = cameras_gt.get_camera_center()
    cameras_pred = PerspectiveCameras(R=R_pred, T=T_pred)
    cc_pred = cameras_pred.get_camera_center()

    cc_hat, scale, R_offset, t_offset = compute_optimal_alignment(cc_gt, cc_pred)
    norm = torch.linalg.norm(cc_gt - cc_hat, dim=1) / gt_scene_scale

    norms = np.ndarray.tolist(norm.detach().cpu().numpy())
    
    return norms, cc_hat, R_offset, t_offset,scale


def update_camera_poses(R_pred, T_pred, R, t, scale=None):
    cameras_pred = PerspectiveCameras(R=R_pred, T=T_pred)
    cc_pred = cameras_pred.get_camera_center()
    opt_center = scale * cc_pred @ R + t

    # update the camera pose with the optimized centerm,    
    updated_R = R_pred @ R.cuda().T #(R.cuda() @ R_pred.transpose(1,2)).transpose(1,2)
    updated_t = - torch.matmul(opt_center.cuda().unsqueeze(1), updated_R).squeeze(1)
    # updated_t = (t - opt_center @ R.T).cuda()
    return updated_R, updated_t


def align_rotations_and_translations(R_pred, T_pred, R_gt, T_gt):
    """
    对齐两个旋转和平移序列，并调整尺度。

    参数:
    R_pred: torch.Tensor, 形状为 (N, 3, 3)，预测的旋转矩阵序列。
    T_pred: torch.Tensor, 形状为 (N, 3)，预测的平移向量序列。
    R_gt: torch.Tensor, 形状为 (N, 3, 3)，真实的旋转矩阵序列。
    T_gt: torch.Tensor, 形状为 (N, 3)，真实的平移向量序列。

    返回:
    aligned_R: torch.Tensor, 形状为 (N, 3, 3)，对齐后的预测旋转矩阵序列。
    aligned_T: torch.Tensor, 形状为 (N, 3)，对齐后的预测平移向量序列。
    scale: float，对齐后的尺度因子。
    """

    assert R_pred.shape == R_gt.shape, "预测和真实的旋转矩阵序列形状必须相同"
    assert T_pred.shape == T_gt.shape, "预测和真实的平移向量序列形状必须相同"
    
    N = R_pred.shape[0]
    
    # 计算质心
    pred_centroid = torch.mean(T_pred, dim=0)
    gt_centroid = torch.mean(T_gt, dim=0)
    
    # 去中心化
    pred_centers_centered = T_pred - pred_centroid
    gt_centers_centered = T_gt - gt_centroid
    
    # 计算协方差矩阵
    H = torch.matmul(pred_centers_centered.T, gt_centers_centered)
    
    # 使用SVD分解
    U, S, Vt = torch.linalg.svd(H)
    
    # 计算最优旋转矩阵
    R_align = torch.matmul(Vt.T, U.T)
    
    # 确保旋转矩阵是正交的，防止反射
    if torch.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = torch.matmul(Vt.T, U.T)
    
    # 计算尺度因子
    scale = torch.sum(S) / torch.sum(pred_centers_centered ** 2)
    
    # 计算最优平移向量
    T_align = gt_centroid - scale * torch.matmul(R_align, pred_centroid)
    
    # 对齐旋转、平移和尺度
    aligned_R = torch.zeros_like(R_pred)
    aligned_T = torch.zeros_like(T_pred)
    for i in range(N):
        aligned_R[i] = torch.matmul(R_align, R_pred[i])
        aligned_T[i] = scale * torch.matmul(R_align, T_pred[i]) + T_align

    return aligned_R, aligned_T, scale

