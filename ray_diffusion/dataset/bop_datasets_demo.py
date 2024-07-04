import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import open3d

from ray_diffusion.utils.bbox import mask_to_bbox
from ray_diffusion.utils.bop_utils import load_depth_points, visualize_object_model_with_depth_points

def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.
    the xyxy format is used for the bounding box. specifically, the bounding box is represented as [x1, y1, x2, y2]. 
    where x1 is the min column, y1 is the min row, x2 is the max column, and y2 is the max row.
    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,). 
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox

def generate_samples():
    # evenly sample 30 images from the list as test set
    img_ids = range(1312)
    
    train_list = []
    test_list = []
    template_list = []

    # evenly sample 8 images as template images from the img ids
    template_list = [89,342,428,673,784,1293,890,1220]

    # select 30 images as test set, and the rest as training set, excluding the template images
    for i in img_ids:
        if i not in template_list:
            if i % 30 == 0:
                test_list.append(i)
            else:
                train_list.append(i)

    # visualize all the template images in one figure
    img_path = "/home/hjw/datasets/bop-dataset/lm/lm_train/train/000006/rgb/"

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i, img_id in enumerate(template_list):
        img = Image.open(osp.join(img_path, f"{img_id:06d}.png"))
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].set_title(f"Template {i}")
    plt.show()

    return train_list, test_list, template_list



class BOPDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir=None,
        bboxes=None,
        mask_images=False,
    ):
        """
        Dataset for custom images. If mask_dir is provided, bounding boxes are extracted
        from the masks. Otherwise, bboxes must be provided.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_images = mask_images
        self.bboxes = []
        self.images = []
        self.mask_image_paths = []
        self.gts = []
        self.train_list, self.test_list, self.template_list = generate_samples()

        if mask_images:
            for image_name, mask_name in tqdm(
                zip(sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir)))
            ):
                image = Image.open(osp.join(image_dir, image_name))
                mask = Image.open(osp.join(mask_dir, mask_name)).convert("L")
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                if mask.size != image.size:
                    mask = mask.resize(image.size)
                mask = Image.fromarray(np.array(mask) > 125)
                image = Image.composite(image, white_image, mask)
                self.images.append(image)
                self.gts.append(self.get_gt_data(osp.join(mask_dir, mask_name), 
                                                 "/home/hjw/datasets/bop-dataset/lm/train/000006/scene_camera.json",
                                                  "/home/hjw/datasets/bop-testset/lm/train/000006/scene_gt.json",
                                                  "/home/hjw/datasets/bop-testset/lm/train/000006/scene_gt_info.json"))
        else:
            for image_path in sorted(os.listdir(image_dir)):
                self.images.append(Image.open(osp.join(image_dir, image_path)))
        self.n = len(self.images)
        if bboxes is None:
            for mask_path in sorted(os.listdir(mask_dir))[: self.n]:
                mask = plt.imread(osp.join(mask_dir, mask_path))
                if len(mask.shape) == 3:
                    mask = mask[:, :, :3]
                else:
                    mask = np.dstack([mask, mask, mask])
                self.bboxes.append(mask_to_bbox(mask))
        else:
            self.bboxes = bboxes
        self.jitter_scale = [1.15, 1.15]
        self.jitter_trans = [0, 0]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return 1

    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32))
        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop
    
    def get_gt_data(self, mask_path, cam_file, gt_file,gt_info_file):
        mask_file_name = osp.basename(mask_path)
        img_id = int(mask_file_name.split("_")[0])
        mask_id = int(mask_file_name.split("_")[1].split(".")[0])

        with open(cam_file) as f:
            cams = json.load(f)
        with open(gt_file) as f:
            gts = json.load(f)
        with open(gt_info_file) as f:
            gt_info = json.load(f)

        cam_K = cams[str(img_id)]["cam_K"]
        depth_scale = cams[str(img_id)]["depth_scale"] 
        t_ = gts[str(img_id)][mask_id]["cam_t_m2c"]
        R_ = gts[str(img_id)][mask_id]["cam_R_m2c"]
        bbox = gt_info[str(img_id)][mask_id]["bbox_obj"]
        t = np.array(t_)/1000
        R = np.array(R_).reshape(3, 3)
        
        obj_id = gts[str(img_id)][mask_id]["obj_id"]

        # load the object model with Open3D
        obj_model = open3d.io.read_triangle_mesh("/home/hjw/datasets/bop-testset/lmo/models_eval/obj_{:06d}.ply".format(obj_id))
        # sample 5000 points on the object model
        obj_pcd = np.array(obj_model.sample_points_poisson_disk(5000).points) / 1000

       
        obj_depth_pcd = load_depth_points(img_id, mask_id,np.array(cam_K).reshape(3, 3), bbox, depth_scale=depth_scale*1000.0)
        
        # visualize the object model in the camera coordinate system with its depth points with open3d
        # visualize_object_model_with_depth_points(obj_pcd_cam, obj_depth_pcd, R, t)

        return {"R": R, "t": t, "obj_id": obj_id, "cam_K": cam_K, "obj_pcd": obj_pcd, "obj_depth_pcd": obj_depth_pcd}

    def __getitem__(self, index):
        return self.get_data()

    def get_data(self, ids=(0, 1, 2, 3, 4, 5)): # take 6 images at a time
        images = [self.images[i] for i in ids]
        bboxes = [self.bboxes[i] for i in ids]

        images_transformed = []
        crop_parameters = []
        for _, (bbox, image) in enumerate(zip(bboxes, images)):
            w, h = image.width, image.height
            bbox = np.array(bbox)
            bbox_jitter = self._jitter_bbox(bbox)
            image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)
            images_transformed.append(self.transform(image))
            crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            length = max(w, h)
            s = length / min(w, h)
            cc = s - 2 * s * crop_center / length
            crop_width = 2 * s * (bbox[2] - bbox[0]) / length
            crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])

            crop_parameters.append(crop_params.float())
        images = images_transformed

        batch = {}
        batch["image"] = torch.stack(images)
        batch["n"] = len(images)
        batch["crop_params"] = torch.stack(crop_parameters)
        batch["R"] = torch.stack([torch.tensor(self.gts[i]["R"]).float() for i in ids])
        batch["T"] = torch.stack([torch.tensor(self.gts[i]["t"]).float() for i in ids])
        batch["obj_ids"] = [self.gts[i]["obj_id"] for i in ids]
        batch["cam_Ks"] = [self.gts[i]["cam_K"] for i in ids]
        batch["obj_pcd"] = [self.gts[i]["obj_pcd"] for i in ids]
        batch["obj_depth_pcd"] = [self.gts[i]["obj_depth_pcd"] for i in ids]
    

        return batch




