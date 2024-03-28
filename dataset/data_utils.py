import numpy as np
from utils.data_process import DataProcessing as DP
from pathlib import Path
from os.path import join
import numpy as np
import pickle
import os
from sklearn.neighbors import KDTree


def get_sk_DownSampled_data(pc_name, dataset_path, remap_lut, data_name):

    seq_id, frame_id = pc_name[0], pc_name[1]

    point_path = join(dataset_path, seq_id, 'velodyne', frame_id + '.bin')
    label_path = join(dataset_path, seq_id, 'labels', frame_id + '.label')
    
    scan = np.fromfile(point_path, dtype=np.float32)
    # if data_name == 'nuScenes':
    #     scan = scan.reshape((-1, 3))
    #     points = scan[:, 0:3]  # get xyz
    #     remissions = np.ones(len(points))
    # else:
    scan = scan.reshape((-1, 4))
    # put in attribute
    points = scan[:, 0:3]  # get xyz
    remissions = scan[:, 3]  # get remission

    label = np.fromfile(label_path, dtype=np.int32)
    label = label.reshape((-1))
   
    return points, remissions, label

def get_sk_data(pc_name, dataset_path, remap_lut, data_name):

    seq_id, frame_id = pc_name[0], pc_name[1]

    point_path = join(dataset_path, seq_id, 'velodyne', frame_id + '.bin')
    label_path = join(dataset_path, seq_id, 'labels', frame_id + '.label')
    
    scan = np.fromfile(point_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # put in attribute
    points = scan[:, 0:3]  # get xyz
    remissions = scan[:, 3]  # get remission

    # load labels
    label = np.fromfile(label_path, dtype=np.int32)
    label = label.reshape((-1))
    
    if data_name == 'SemanticKITTI' or data_name == 'SemanticPOSS':
        label = label & 0xFFFF  # semantic label in lower half
    if remap_lut is not None:
        label = remap_lut[label]

    # read pkl with search tree
    # with open(kd_tree_path, 'rb') as f:
    #     search_tree = pickle.load(f)
    # search_tree = KDTree(points)search_tree

    return points, remissions, label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def augment_scale(points):  # from xmuda
    s_min = 0.95
    s_max = 1.05
    s = (s_max - s_min) * np.random.random() + s_min
    # rot_matrix = np.eye(3, dtype=np.float32)
    # theta = np.random.rand() * rot_z
    # z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                             [np.sin(theta), np.cos(theta), 0],
    #                             [0, 0, 1]], dtype=np.float32)
    # rot_matrix = rot_matrix.dot(z_rot_matrix)
    points = points * s

    return points


def get_Kitti_like_Nuscenes_data(point_path, label_path, kd_tree_path, remap_lut):

    scan = np.fromfile(point_path, dtype=np.float32)
    points = scan.reshape((-1, 3))
    # put in attribute
    # points = scan[:, 0:3]  # get xyz
    remissions = np.zeros(points.shape[0])
    # load labels
    label = np.fromfile(label_path, dtype=np.uint8)
    label = label.reshape((-1)).astype(np.int32)
    # label = label & 0xFFFF  # semantic label in lower half
    if remap_lut is not None:
        label = remap_lut[label]
    # read pkl with search tree
    with open(kd_tree_path, 'rb') as f:
        search_tree = pickle.load(f)
    return points, remissions, search_tree, label


def get_poss_data(point_path, label_path, kd_tree_path, remap_lut):

    scan = np.fromfile(point_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # put in attribute
    points = scan[:, 0:3]  # get xyz
    remissions = scan[:, 3]  # get remission
    # if all goes well, open label
    label = np.fromfile(label_path, dtype=np.int32)
    label = label.reshape((-1))
    label = label & 0xFFFF  # semantic label in lower half
    label = remap_lut[label]
    # read pkl with search tree
    with open(kd_tree_path, 'rb') as f:
        search_tree = pickle.load(f)

    return points, remissions, search_tree, label


def get_data(point_path, label_path, kd_tree_path, remap_lut=None):

    scan = np.fromfile(point_path, dtype=np.float32)
    points = scan.reshape((-1, 3))

    # load labels
    label = np.fromfile(label_path, dtype=np.uint8)
    label = label.reshape((-1)).astype(np.int32)

    remissions = np.zeros_like(label)  # get remission

    # read pkl with search tree
    with open(kd_tree_path, 'rb') as f:
        search_tree = pickle.load(f)

    return points, remissions, search_tree, label



def range_projection(points, remissions, labels, proj_fov_up, proj_fov_down, proj_H, proj_W):
    """
                Project a pointcloud into a spherical projection image.projection.
                Function takes no arguments because it can be also called externally
                if the value of the constructor was not set (in case you change your
                mind about wanting the projection)
    """
    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi  # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y = np.copy(proj_y)  # stope a copy in original order

    # # copy of depth in original order
    # unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    remission = remissions[order]
    labels = labels[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.zeros((proj_H, proj_W), dtype=np.float32)
    proj_range[proj_y, proj_x] = depth
    proj_xyz = np.zeros((proj_H, proj_W, 3), dtype=np.float32)
    proj_xyz[proj_y, proj_x] = points
    proj_remission = np.zeros((proj_H, proj_W), dtype=np.float32)
    proj_remission[proj_y, proj_x] = remission
    proj_idx = np.zeros((proj_H, proj_W), dtype=np.int32)
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx > 0).astype(np.int32)
    proj_labels = np.zeros((proj_H, proj_W), dtype=np.int32)
    proj_labels[proj_y, proj_x] = labels

    proj = np.concatenate((proj_range[np.newaxis, ...],
                           #    np.transpose(proj_xyz, (2, 0, 1)), 不要绝对位置
                           proj_remission[np.newaxis, ...]), axis=0)
    return points, remissions, labels, proj, proj_labels, proj_mask, proj_y, proj_x


def augment_noisy_rot(points, noisy_rot=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    rot_matrix += np.random.randn(3, 3) * noisy_rot
    points = points.dot(rot_matrix)

    return points


def augment_flip_x(points, flip_x=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
    points = points.dot(rot_matrix)

    return points


def augment_flip_y(points, flip_y=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
    points = points.dot(rot_matrix)

    return points


def augment_rot_z(points, rot_z=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    theta = np.random.rand() * rot_z
    z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]], dtype=np.float32)
    rot_matrix = rot_matrix.dot(z_rot_matrix)
    points = points.dot(rot_matrix)

    return points


def augment_instance(points, labels, remissions, file_list, remap_lut):  # from xmuda

    pick_inst = np.random.choice(len(file_list), 20, replace=False)
    # gound_ind = np.where(labels == 9 or labels == 10 or labels==12)
    gound_ind = np.where(labels == 1)
    if len(gound_ind[0]) == 0:
        return points, labels, remissions
    for inst in pick_inst:
        # 找到一个ground点 9 10 12 为ground类
        pick_ground = np.random.choice(len(gound_ind[0]), 1, replace=False)
        gound_point = points[pick_ground, :]
        # load instance pc
        filename = file_list[inst]
        inst_pc = np.fromfile(filename, dtype=np.float32)
        inst_pc = inst_pc.reshape((-1, 4))
        # 这里加上地面坐标，相当于粘贴到这个位置
        inst_pc[:, 0:3] += gound_point
        points = np.concatenate((points, inst_pc[:, 0:3]), axis=0)
        remissions = np.concatenate((remissions, inst_pc[:, -1]), axis=0)
        # load instance pc label
        lb_filename = filename.replace(
            'velodyne', 'labels').replace('.bin', '.label')
        inst_lab = np.fromfile(lb_filename, dtype=np.int32)
        inst_lab = inst_lab.reshape((-1))
        inst_lab = inst_lab & 0xFFFF  # semantic label in lower half
        inst_lab = remap_lut[inst_lab]
        labels = np.concatenate((labels, inst_lab), axis=0)

        assert labels.shape[0] == remissions.shape[0] == points.shape[0], "Don't have same number"

    if points.shape[0] > 140000:
        pick_idx = np.random.choice(
            points.shape[0], 14_0000, replace=False)
        points = points[pick_idx, :]
        labels = labels[pick_idx]
        remissions = remissions[pick_idx]

    return points, labels, remissions
