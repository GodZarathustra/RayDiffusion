import argparse
import base64
import io
import json
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from ray_diffusion.dataset import CustomDataset
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras
from ray_diffusion.utils.visualization import view_color_coded_images_from_tensor

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

def load_ground_truth_cameras(mask_path="000003_000001.png"):
    cam_file = "/home/hjw/datasets/bop-testset/lmo/test/000002/scene_camera.json"
    gt_file = "/home/hjw/datasets/bop-testset/lmo/test/000002/scene_gt.json"
    mask_file_name = osp.basename(mask_path)
    img_id = int(mask_file_name.split("_")[0])
    mask_id = int(mask_file_name.split("_")[1].split(".")[0])

    with open(cam_file) as f:
        cams = json.load(f)
    with open(gt_file) as f:
        gts = json.load(f)
    
    cam = cams[str(img_id)]
    t = gts[str(img_id)][mask_id]["cam_t_m2c"]
    R = gts[str(img_id)][mask_id]["cam_R_m2c"]
    return R, t

def plotly_scene_visualization(R_pred, T_pred):
    num_frames = len(R_pred)

    camera = {}
    for i in range(num_frames):
        camera[i] = PerspectiveCameras(R=R_pred[i, None], T=T_pred[i, None])

    fig = plot_scene(
        {"scene": camera},
        camera_scale=0.03,
    )
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")
    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
    return fig


def main(image_dir, model_dir, mask_dir, bbox_path, output_path):
    device = torch.device("cuda:0")
    model, cfg = load_model(model_dir, device=device)
    if osp.exists(bbox_path):
        bboxes = json.load(open(bbox_path))
    else:
        bboxes = None
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        bboxes=bboxes,
        mask_images=False,
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
    print("R: ", predicted_cameras.R)
    print("T: ", predicted_cameras.T)
    # Visualize cropped and resized images
    fig = plotly_scene_visualization(predicted_cameras.R, predicted_cameras.T)
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



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
    R,t = load_ground_truth_cameras()
    print("R: ", R)
    print("t: ", t)