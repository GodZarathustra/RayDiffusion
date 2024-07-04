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

from pytorch3d.renderer import PerspectiveCameras
from ray_diffusion.utils.bbox import mask_to_bbox
from ray_diffusion.utils.bop_utils import load_depth_points, visualize_object_model_with_depth_points


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
    # img_path = "/home/hjw/datasets/bop-dataset/lm/lm_train/train/000006/rgb/"

    # fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    # axs = axs.flatten()
    # for i, img_id in enumerate(template_list):
    #     img = Image.open(osp.join(img_path, f"{img_id:06d}.png"))
    #     axs[i].imshow(img)
    #     axs[i].axis("off")
    #     axs[i].set_title(f"Template {i}")
    # plt.show()

    return train_list, test_list, template_list


BOP_DIR = "/home/hjw/datasets/bop-dataset"
TRAIN_LS, TEST_LS, TEMPLATE_LS = generate_samples()

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



def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def _transform_intrinsic(image, bbox, principal_point, focal_length):
    # Rescale intrinsics to match bbox
    half_box = np.array([image.width, image.height]).astype(np.float32) / 2
    org_scale = min(half_box).astype(np.float32)

    # Pixel coordinates
    principal_point_px = half_box - (np.array(principal_point) * org_scale)
    focal_length_px = np.array(focal_length) * org_scale
    principal_point_px -= bbox[:2]
    new_bbox = (bbox[2:] - bbox[:2]) / 2
    new_scale = min(new_bbox)

    # NDC coordinates
    new_principal_ndc = (new_bbox - principal_point_px) / new_scale
    new_focal_ndc = focal_length_px / new_scale

    principal_point = torch.tensor(new_principal_ndc.astype(np.float32))
    focal_length = torch.tensor(new_focal_ndc.astype(np.float32))

    return principal_point, focal_length


def construct_camera_from_batch(batch, device):
    if isinstance(device, int):
        device = f"cuda:{device}"

    return PerspectiveCameras(
        R=batch["R"].reshape(-1, 3, 3),
        T=batch["T"].reshape(-1, 3),
        focal_length=batch["focal_lengths"].reshape(-1, 2),
        principal_point=batch["principal_points"].reshape(-1, 2),
        image_size=batch["image_sizes"].reshape(-1, 2),
        device=device,
    )


def save_batch_images(images, fname):
    cmap = plt.get_cmap("hsv")
    num_frames = len(images)
    num_rows = len(images)
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows):
        for j in range(4):
            if i < num_frames:
                axs[i * 4 + j].imshow(unnormalize_image(images[i][j]))
                for s in ["bottom", "top", "left", "right"]:
                    axs[i * 4 + j].spines[s].set_color(cmap(i / (num_frames)))
                    axs[i * 4 + j].spines[s].set_linewidth(5)
                axs[i * 4 + j].set_xticks([])
                axs[i * 4 + j].set_yticks([])
            else:
                axs[i * 4 + j].axis("off")
    plt.tight_layout()
    plt.savefig(fname)


def jitter_bbox(square_bbox, jitter_scale=(1.1, 1.2), jitter_trans=(-0.07, 0.07)):
    square_bbox = np.array(square_bbox.astype(float))
    s = np.random.uniform(jitter_scale[0], jitter_scale[1])
    tx, ty = np.random.uniform(jitter_trans[0], jitter_trans[1], size=2)
    side_length = square_bbox[2] - square_bbox[0]
    center = (square_bbox[:2] + square_bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s
    ul = center - extent
    lr = ul + 2 * extent
    return np.concatenate((ul, lr))


class BOPDataset(Dataset):
    def __init__(
        self,
        data_name,
        split="train",
        transform=None,
        num_images=2,
        img_size=224,
        mask_images=False,
        crop_images=True,
        apply_augmentation=True,
        normalize_cameras=True,
        no_images=False,
        seed=0,
        load_extra_cameras=False,
        
    ):
        """
        Dataset for custom images. If mask_dir is provided, bounding boxes are extracted
        from the masks. Otherwise, bboxes must be provided.
        """
        self.image_dir = os.path.join(BOP_DIR, data_name, split, "000006", "rgb")
        self.mask_dir = self.image_dir.replace("rgb", "mask_visib")
        self.mask_images = mask_images
        self.num_images = num_images
        
        self.jitter_scale = [1.15, 1.15]
        self.jitter_trans = [0, 0]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.metadata = self.load_meta_data(6)
    
    def __len__(self):
        return len(TRAIN_LS) if self.split == "train" else len(TEST_LS)

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
    
    def load_meta_data(self, obj_id):
        cam_file = "/home/hjw/datasets/bop-testset/lm/train/000006/scene_camera.json"
        gt_file = "/home/hjw/datasets/bop-testset/lm/train/000006/scene_gt.json"
        gt_info_file = "/home/hjw/datasets/bop-testset/lm/train/000006/scene_gt_info.json"

        with open(cam_file) as f:
            cams = json.load(f)
        with open(gt_file) as f:
            gts = json.load(f)
        with open(gt_info_file) as f:
            gt_info = json.load(f)

        # load the object model with Open3D
        obj_model = open3d.io.read_triangle_mesh("/home/hjw/datasets/bop-testset/lmo/models_eval/obj_{:06d}.ply".format(obj_id))
        # sample 5000 points on the object model
        obj_pcd = np.array(obj_model.sample_points_poisson_disk(5000).points) / 1000

        return {"cams": cams, "gts": gts, "gt_info": gt_info, "obj_pcd": obj_pcd}

    def load_input_data(self, img_id_ls, cams, gts, gt_info, no_images=False):
        # Read image & camera information from annotations
        annos = []
        images = []
        image_sizes = []
        PP = [] # principal point
        FL = [] # focal length
        crop_parameters = []
        filenames = []

        for img_id in img_id_ls:
            for ins_id in gts[str(img_id)]:
                if not no_images:
                    image = Image.open(osp.join(self.image_dir, f"{img_id:06d}.png"))

                    # Optionally mask images with black background
                    if self.mask_images:
                        black_image = Image.new("RGB", image.size, (0, 0, 0))
                        mask_name = osp.basename(f"{img_id:06d}_{ins_id:06d}.png")
                        mask_path = osp.join(self.mask_dir, mask_name)
                        mask = Image.open(mask_path).convert("L")

                        if mask.size != image.size:
                            mask = mask.resize(image.size)
                        mask = Image.fromarray(np.array(mask) > 125)
                        image = Image.composite(image, black_image, mask)

                    # Determine crop, Resnet wants square images
                    bbox_init = (
                        gt_info[str(img_id)][ins_id]["bbox_obj"] # TODO check if this is the correct bbox
                        if self.crop_images
                        else [0, 0, image.width, image.height]
                    )
                    bbox = square_bbox(np.array(bbox_init))
                    if self.apply_augmentation:
                        bbox = jitter_bbox(
                            bbox,
                            jitter_scale=self.jitter_scale,
                            jitter_trans=self.jitter_trans,
                        )
                    bbox = np.around(bbox).astype(int)

                    # Crop parameters
                    crop_center = (bbox[:2] + bbox[2:]) / 2
                    # convert crop center to correspond to a "square" image
                    width, height = image.size
                    length = max(width, height)
                    s = length / min(width, height)
                    crop_center = crop_center + (length - np.array([width, height])) / 2
                    # convert to NDC
                    cc = s - 2 * s * crop_center / length
                    crop_width = 2 * s * (bbox[2] - bbox[0]) / length
                    crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])

                    # calculate principal point and focal length
                    principal_point, focal_length = _transform_intrinsic(
                        image, bbox, cams[str(img_id)]["cam_principal"], cams[str(img_id)]["cam_K"]
                    )

                    # Crop and normalize image
                    image = self._crop_image(image, bbox)
                    image = self.transform(image)
                    images.append(image[:, : self.img_size, : self.img_size])
                    crop_parameters.append(crop_params)

                else:
                    principal_point, focal_length = _transform_intrinsic(
                        image, bbox, cams[str(img_id)]["cam_principal"], cams[str(img_id)]["cam_K"]
                    )

                PP.append(principal_point)
                FL.append(focal_length)
                image_sizes.append(torch.tensor([self.img_size, self.img_size]))
                filenames.append(f"{img_id:06d}_{ins_id:06d}.png")

        if not no_images:
            if self.load_extra_cameras:
                # Remove the extra loaded image, for saving space
                images = images[: self.num_images]

            images = torch.stack(images)
            crop_parameters = torch.stack(crop_parameters)
        else:
            images = None
            crop_parameters = None

        # Assemble batch info to send back
        R = torch.stack([torch.tensor(anno["R"]) for anno in annos])
        T = torch.stack([torch.tensor(anno["T"]) for anno in annos])
        focal_lengths = torch.stack(FL)
        principal_points = torch.stack(PP)
        image_sizes = torch.stack(image_sizes)


        batchdata = {
            "images": images,
            "R": R,
            "T": T,
            "focal_lengths": focal_lengths,
            "principal_points": principal_points,
            "image_sizes": image_sizes,
            "crop_parameters": crop_parameters,
            "filenames": filenames,
        }

        return batchdata
    

    def __getitem__(self, index):

        # Load the template data
        templates = self.load_input_data(TEMPLATE_LS, self.metadata["cams"], self.metadata["gts"], self.metadata["gt_info"], no_images=True)
        
        return self.get_data(index=index, templates=templates)

    def get_data(self, index=None,templates=None, no_images=False):
        
        data_id = TRAIN_LS[index] if self.split == "train" else TEST_LS[index]

        input_data = self.load_input_data([data_id], self.metadata["cams"], self.metadata["gts"], self.metadata["gt_info"], no_images=no_images)

        # concatenate the template data with the input data
        images = torch.cat([templates["images"], input_data["images"]], dim=0)
        R = torch.cat([templates["R"], input_data["R"]], dim=0)
        T = torch.cat([templates["T"], input_data["T"]], dim=0)
        focal_lengths = torch.cat([templates["focal_lengths"], input_data["focal_lengths"]], dim=0)
        principal_points = torch.cat([templates["principal_points"], input_data["principal_points"]], dim=0)
        image_sizes = torch.cat([templates["image_sizes"], input_data["image_sizes"]], dim=0)
        crop_parameters = torch.cat([templates["crop_parameters"], input_data["crop_parameters"]], dim=0)
        filenames = templates["filenames"] + input_data["filenames"]
        

        batch = {
            "n": self.num_images,
            "image": images,
            "R": R,
            "T": T,
            "focal_length": focal_lengths,
            "principal_point": principal_points,
            "image_size": image_sizes,
            "crop_parameters": crop_parameters,
            "filename": filenames,
        }

        return batch




