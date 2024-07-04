import argparse
import base64
import io
import json
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np
import plotly
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    MeshRenderer,
    SoftPhongShader,
)

from ray_diffusion.dataset import BOPDataset
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras
from ray_diffusion.utils.visualization import view_color_coded_images_from_tensor
from ray_diffusion.utils.bop_utils import visualize_object_model_with_depth_points
from ray_diffusion.eval.utils import full_scene_scale , update_camera_poses, align_rotations_and_translations
from ray_diffusion.eval.eval_category import compute_angular_error_batch
from ray_diffusion.eval.eval_category import compute_camera_center_error
from ray_diffusion.eval.eval_category import n_to_np_rotations




HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
<body><img src="data:image/png;charset=utf-8;base64,{image_encoded}"/>
{plotly_html}</body></html>"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="examples/lmo/images")
    parser.add_argument("--model_dir", type=str, default="models/co3d_diffusion")
    parser.add_argument("--mask_dir", type=str, default="examples/lmo/masks")
    parser.add_argument("--bbox_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="lmo_output_cameras.html")
    return parser

def create_plotly_cameras_visualization(cameras_gt, cameras_pred, num):
    num_frames = cameras_gt.R.shape[0]
    name = f"Vis {num} GT vs Pred Cameras"
    camera_scale = 0.05
    cmap = plt.get_cmap("hsv")

    # Cameras_pred is already a 2D list of unbatched cameras
    # But cameras_gt is a 1D list of batched cameras
    scenes = {f"Vis {num} GT vs Pred Cameras": {}}
    for i in range(num_frames):
        scenes[name][f"Pred Camera {i}"] = PerspectiveCameras(
            R=cameras_pred[i].R, T=cameras_pred[i].T
        )
    for i in range(num_frames):
        scenes[name][f"GT Camera {i}"] = PerspectiveCameras(
            R=cameras_gt[i].R, T=cameras_gt[i].T
        )

    fig = plot_scene(
        scenes,
        camera_scale=camera_scale,
    )
    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=800, width=800)

    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
        fig.data[i].line.width = 4
        fig.data[i + num_frames].line.dash = "dash"
        fig.data[i + num_frames].line.color = matplotlib.colors.to_hex(
            cmap(i / (num_frames))
        )
        fig.data[i + num_frames].line.width = 4

    return fig

def main(image_dir, model_dir, mask_dir, bbox_path, output_path):
    device = torch.device("cuda:0")
    model, cfg = load_model(model_dir, device=device)
    if osp.exists(bbox_path):
        bboxes = json.load(open(bbox_path))
    else:
        bboxes = None
    dataset = BOPDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        bboxes=bboxes,
        mask_images=True,
    )
    num_frames = dataset.n
    batch = dataset.get_data(ids=np.arange(num_frames))
    images = batch["image"].to(device)
    crop_params = batch["crop_params"].to(device)

    is_regression = cfg.training.regression
    if is_regression:
        # regression
        pred = predict_cameras(
            model=model,
            images=images,
            device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            use_regression=True,
        )
        predicted_cameras = pred[0]
    else:
        # diffusion
        pred = predict_cameras(
            model=model,
            images=images,
            device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            additional_timesteps=(70,),  # We found that X0 at T=30 is best.
            rescale_noise="zero",
            use_regression=False,
            max_num_images=None if num_frames <= 8 else 8,  # Auto-batch for N > 8.
            pbar=True,
        )
        predicted_cameras = pred[1][0]
    
    # calculate the ADD metric, as the predicted R and T are already in the camera coordinate system(transforms world coordinate to camera coordinate), 
    # we can directly compare the object model with the GT object model
    # ADD = []
    # GT_Rs = [torch.tensor(batch["R"][i]).to(device).float() for i in range(num_frames)]
    # GT_Ts = [torch.tensor(batch["T"][i]).to(device).float() for i in range(num_frames)]
    # obj_pcd = [torch.tensor(batch["obj_pcd"][i]).to(device).float() for i in range(num_frames)]
    # obj_depth_pcd = [torch.tensor(batch["obj_depth_pcd"][i]).to(device).float() for i in range(num_frames)]
   
    # for i in range(num_frames):
    #     obj_pcd_cam = obj_pcd[i] @ predicted_cameras.R[i].T + predicted_cameras.T[i]
    #     obj_pcd_gt = obj_pcd[i] @ GT_Rs[i].T + GT_Ts[i]
    #     dist = torch.mean(torch.norm(obj_pcd_cam - obj_pcd_gt, dim=1))
    #     ADD.append(dist)

    # print("ADD metric: ", ADD)
    # print("Mean ADD: ", torch.mean(torch.stack(ADD)))


    # calculate the scene scale, camera center error, and angular error

    # R_gt = batch["R"].to(device).transpose(1, 2)
    # T_gt = - torch.matmul(batch["T"].to(device).unsqueeze(1), R_gt).squeeze(1)

    # batch["R"] = R_gt
    # batch["T"] = T_gt

    R_gt = batch["R"].to(device)
    T_gt = batch["T"].to(device)


    gt_scene_scale = full_scene_scale(batch)

    R_pred = predicted_cameras.R
    T_pred = predicted_cameras.T

    R_pred_rel = n_to_np_rotations(num_frames, R_pred).cpu().numpy()
    R_gt_rel = n_to_np_rotations(num_frames, R_gt).cpu().numpy()
    R_error = compute_angular_error_batch(R_pred_rel, R_gt_rel)
    camera_center_errors, cc_optimized, R_off, T_off, scale = compute_camera_center_error(
        R_pred, T_pred, R_gt, T_gt, gt_scene_scale
    )

    print("R error: ", R_error)
    print("Camera center error: ", camera_center_errors)
    print("Mean camera center error: ", np.mean(camera_center_errors))
    print("Mean R error: ", np.mean(R_error))

    # convert the optimized camera center to camera coordinate system
    # optimized_center = optimized_center.to(device).unsqueeze(1)
    # optimized_T = - torch.matmul(optimized_center, R_pred).squeeze(1)

    # R_optimized, T_optimized = update_camera_poses(R_pred, T_pred, R_off, T_off, scale)
    R_opt, T_opt , scale = align_rotations_and_translations(R_pred, T_pred, R_gt, T_gt)
    mean_cam_center_error = torch.mean(torch.linalg.norm(T_opt - T_gt, dim=1))
    R_pred_rel = n_to_np_rotations(num_frames, R_opt).cpu().numpy()
    R_gt_rel = n_to_np_rotations(num_frames, R_gt).cpu().numpy()
    R_error = compute_angular_error_batch(R_pred_rel, R_gt_rel)
    print("Mean optimized camera center error: ", mean_cam_center_error)
    print("Mean optimized R error: ", np.mean(R_error))


    visualize_cameras(R_opt, T_opt, R_gt, T_gt)

    # visualize the object model in the camera coordinate system with its depth points with open3d
    # for i in range(num_frames):
    #     visualize_object_model_with_depth_points(batch["obj_pcd"][i], batch["obj_depth_pcd"][i], R_opt[i], T_opt[i])


    # Visualize cropped and resized images, predicted cameras, and GT cameras.
    # format the gt cameras with the same shape as the predicted cameras

    cameras_gt = PerspectiveCameras(R=R_gt, T=T_gt)
    cameras_pred = PerspectiveCameras(R=R_opt, T=T_opt)

    fig = create_plotly_cameras_visualization(cameras_gt, cameras_pred, num_frames)
    html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
    s = io.BytesIO()
    view_color_coded_images_from_tensor(images)
    plt.savefig(s, format="png", bbox_inches="tight")
    plt.close()
    image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    with open(output_path, "w") as f:
        s = HTML_TEMPLATE.format(
            image_encoded=image_encoded,
            plotly_html=html_plot,
        )
        f.write(s)

def plot_camera(ax, R, T, color, label, linestyle=None):
    """
    在3D轴上绘制相机位置和方向。

    参数:
    ax: matplotlib.axes._subplots.Axes3DSubplot, 3D轴。
    R: torch.Tensor, 形状为 (3, 3)，相机的旋转矩阵。
    T: torch.Tensor, 形状为 (3,)，相机的平移向量。
    color: str, 绘图颜色。
    label: str, 相机标签。
    """
    # 相机中心
    camera_center = T.cpu().numpy()

    # 相机方向 (以相机光轴为方向)
    camera_direction = R[:, 2].cpu().numpy()

    # 绘制相机中心
    ax.scatter(camera_center[0], camera_center[1], camera_center[2], color=color, label=label)

    # 绘制相机方向
    ax.quiver(camera_center[0], camera_center[1], camera_center[2], 
              camera_direction[0], camera_direction[1], camera_direction[2], 
              length=0.01, color=color,linestyle=linestyle)

def visualize_cameras(R_pred, T_pred, R_gt, T_gt):
    """
    在世界坐标系下可视化相机位置和方向。

    参数:
    R_pred: torch.Tensor, 形状为 (N, 3, 3)，预测的旋转矩阵序列。
    T_pred: torch.Tensor, 形状为 (N, 3)，预测的平移向量序列。
    R_gt: torch.Tensor, 形状为 (N, 3, 3)，真实的旋转矩阵序列。
    T_gt: torch.Tensor, 形状为 (N, 3)，真实的平移向量序列。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap("hsv")

    # 绘制预测的相机
    for i in range(R_pred.shape[0]):
        plot_camera(ax, R_pred[i], T_pred[i], color=cmap(i / R_pred.shape[0]), label='Pred' if i == 0 else "", linestyle='-')

    # 绘制真实的相机
    for i in range(R_gt.shape[0]):
        plot_camera(ax, R_gt[i], T_gt[i], color=cmap(i / R_gt.shape[0]), label='GT' if i == 0 else "", linestyle='--')
    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    ax.legend()
    plt.show()




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))

