from utils.data_process import DataProcessing as DP

from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import yaml
import os

import MinkowskiEngine as ME

from sklearn.neighbors import KDTree

class SemanticBase(torch_data.Dataset):
    def __init__(self, mode, data_list=None):
        pass

    def get_class_weight(self):
        return DP.get_class_weights(self.dataset_path, self.data_list, self.num_classes)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        sel_pc, sel_lab, sel_idx, cloud_ind, proj, proj_labels, proj_mask, proj_y, proj_x = self.spatially_regular_gen(
            item, self.data_list)

        # Normalization
        proj = (proj - np.array(cfg.img_means)
                [:, None, None]) / np.array(cfg.img_stds)[:, None, None]
        sel_pc = self.pc_normalize(sel_pc)
        return sel_pc, sel_lab, sel_idx, cloud_ind, proj, proj_labels, proj_mask, proj_y, proj_x

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def spatially_regular_gen(self, item, data_list):
        pass

    def get_data(self, dataset_path, file_path):
        seq_id = file_path[0]
        frame_id = file_path[1]

        point_path = join(dataset_path, seq_id,
                          'velodyne', frame_id + '.bin')

        scan = np.fromfile(point_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # [x,y,z,intensity]

        # scan = np.fromfile(point_path, dtype=np.float32)
        points = scan[:, 0:3]

        label_path = join(dataset_path, seq_id, 'labels', frame_id + '.label')
        # if all goes well, open label
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        # label = np.fromfile(label_path, dtype=np.uint8)
        # label = label.reshape((-1)).astype(np.int32)

        remissions = scan[:, 3]  # get remission

        # kd_tree_path = join(dataset_path, seq_id,
        #                     'KDTree', frame_id + '.pkl')
        # # read pkl with search tree
        # with open(kd_tree_path, 'rb') as f:
        #     search_tree = pickle.load(f)

        # search_tree = KDTree(points)
        search_tree = None

        return points, remissions, search_tree, label

    def collate_fn(self, batch):
        sp_data = []
        slt_pc, sel_lab, slt_idx, cloud_ind = [], [], [], []
        for i in range(len(batch)):
            slt_pc.append(batch[i][0])
            sel_lab.append(batch[i][1])
            slt_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])
            sp_data.append(batch[i][4])

        # slt_pc = np.stack(slt_pc)
        # sel_lab = np.stack(sel_lab)
        # slt_idx = np.stack(slt_idx)
        cloud_ind = np.stack(cloud_ind)

        inputs = {}
        inputs['cloud_inds'] = torch.from_numpy(cloud_ind).long()
        # inputs['selected_idx'] = torch.from_numpy(slt_idx).long()

        # for sparse data
        coords, feats, labels, inverse_map, unique_map, s_len = list(zip(*sp_data))
        inputs['s_lens'] = torch.from_numpy(np.stack(s_len, 0)).long()
        # Generate batched coordinates
        inputs['coords_mink'] = ME.utils.batched_coordinates(coords)
        # Concatenate all lists
        inputs['feats_mink'] = torch.from_numpy(
            np.concatenate(feats, 0)).float()
        inputs['labels_mink'] = torch.from_numpy(
            np.concatenate(labels, 0)).long()

        list_inverse_map = list(inverse_map)
        list_unique_map = list(unique_map)
        post_len, unique_len = 0, 0
        for i_list in range(len(list_inverse_map)):
            list_inverse_map[i_list] = list_inverse_map[i_list] + post_len
            post_len += unique_map[i_list].shape[0]

            list_unique_map[i_list] = list_unique_map[i_list] + unique_len
            unique_len += inverse_map[i_list].shape[0]

        inputs['inverse_map'] = torch.from_numpy(
            np.concatenate(list_inverse_map, 0)).long()
        inputs['unique_map'] = torch.from_numpy(
            np.concatenate(list_unique_map, 0)).long()

        return inputs

    def augment_noisy_rot(self, points, noisy_rot=0.0):  # from xmuda
        rot_matrix = np.eye(3, dtype=np.float32)
        rot_matrix += np.random.randn(3, 3) * noisy_rot
        points = points.dot(rot_matrix)

        return points

    def augment_flip_x(self, points, flip_x=0.0):  # from xmuda
        rot_matrix = np.eye(3, dtype=np.float32)
        rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
        points = points.dot(rot_matrix)

        return points

    def augment_flip_y(self, points, flip_y=0.0):  # from xmuda
        rot_matrix = np.eye(3, dtype=np.float32)
        rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
        points = points.dot(rot_matrix)

        return points

    def augment_rot_z(self, points, rot_z=0.0):  # from xmuda
        rot_matrix = np.eye(3, dtype=np.float32)
        theta = np.random.rand() * rot_z
        z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]], dtype=np.float32)
        rot_matrix = rot_matrix.dot(z_rot_matrix)
        points = points.dot(rot_matrix)

        return points
