import numpy as np
import PIL.Image as Image
import numpy.ma as ma
import open3d
import torch

def load_depth_points(img_id,mask_id, cam_k, obj_bbx, depth_scale=1000.0):
    depth_path = "/home/hjw/datasets/bop-testset/lmo/test/000002/depth/"+ str(img_id).zfill(6) + ".png"
    mask_path = "/home/hjw/datasets/bop-testset/lmo/test/000002/mask_visib/"+ str(img_id).zfill(6) + "_" + str(mask_id).zfill(6) + ".png"

    depth = np.array(Image.open(depth_path))
    vis_mask = np.array(Image.open(mask_path))
    
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(vis_mask, np.array(255)))
    mask = mask_label * mask_depth

    im_height = depth.shape[0]
    im_width = depth.shape[1]
    xmap = np.array([[j for i in range(im_width)] for j in range(im_height)])
    ymap = np.array([[i for i in range(im_width)] for j in range(im_height)])

    rmin, rmax, cmin, cmax = obj_bbx[1], obj_bbx[1] + obj_bbx[3], obj_bbx[0], obj_bbx[0] + obj_bbx[2]

    cam_cx = cam_k[0, 2]
    cam_cy = cam_k[1, 2]
    cam_fx = cam_k[0, 0]
    cam_fy = cam_k[1, 1]

    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xymap_masked = np.concatenate((xmap_masked, ymap_masked), axis=1)
    pt2 = depth_masked / depth_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    depth_pcd = np.concatenate((pt0, pt1, pt2), axis=1) 

    return depth_pcd

def visualize_object_model_with_depth_points(obj_pcd, obj_depth_pcd, R,t):
    
    # transform the object model to the camera coordinate system
    obj_pcd_cam = obj_pcd @ R.T.cpu().numpy() + t.cpu().numpy()

    # check if the data is tensor or numpy array, and convert to numpy array
    if isinstance(obj_pcd_cam, torch.Tensor):
        obj_pcd_cam = obj_pcd_cam.cpu().numpy()
    if isinstance(obj_depth_pcd, torch.Tensor):
        obj_depth_pcd = obj_depth_pcd.cpu().numpy()

    obj_pcd_cam_vis = open3d.geometry.PointCloud()
    obj_pcd_cam_vis.points = open3d.utility.Vector3dVector(obj_pcd_cam)
    obj_depth_pcd_vis = open3d.geometry.PointCloud()
    obj_depth_pcd_vis.points = open3d.utility.Vector3dVector(obj_depth_pcd)
    open3d.visualization.draw_geometries([obj_pcd_cam_vis, obj_depth_pcd_vis])