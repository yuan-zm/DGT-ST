from utils.data_process import DataProcessing as DP

from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import yaml
import os
from .semkitti_base import SemanticBase
import MinkowskiEngine as ME
# slt --> selected

import copy

class SemanticKITTI_dataset(SemanticBase):
    def __init__(self, cfg, mode, data_list=None):
        self.cfg = cfg
        
        self.name = cfg.DATASET_SOURCE.TYPE
        self.dataset_path = os.path.expanduser(cfg.DATASET_SOURCE.DATASET_DIR)

        DATA = yaml.safe_load(open(cfg.DATASET_SOURCE.mapping_file, 'r'))

        if cfg.DATASET_TARGET.TYPE == 'nuScenes_10':
            remap_dict = DATA["sk2nusc_learning_map_10"]
            max_key = max(remap_dict.keys())
            self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

            self.label_name = DATA["sk2nusc_labels_10"]

        self.num_classes = cfg.MODEL_G.NUM_CLASSES
        self.ignored_labels = np.sort([0])

        # for augmentation
        self.noisy_rot = 0.1
        self.flip_y = 0.5
        self.rot_z = 6.2831  # 2 * pi

        # for sparse
        self.quantization_size = cfg.DATASET_SOURCE.VOXEL_SIZE

        self.mode = mode
        if data_list is None:
            if mode == 'training':
                seq_list = ['00', '01', '02', '03', '04', 
                            '05', '06', '07', '09', '10']
                self.data_list = DP.get_file_list(self.dataset_path, seq_list)
            elif mode == 'validation':
                seq_list = ['08']
                self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)

        print("This dataset has:{} files.".format(len(self.data_list)))

    def __len__(self):
        if self.cfg.TRAIN.DEBUG:
            return 30
        else:
            return len(self.data_list)

    def __getitem__(self, item):
        slt_pc, sel_lab, feature, slt_idx, cloud_ind, sp_data = self.spatially_regular_gen(item, self.data_list)

        return slt_pc, sel_lab, slt_idx, cloud_ind, sp_data

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]
        
        pc, remis, tree, labels = self.get_data(self.dataset_path, pc_path)  # get data

        labels = labels & 0xFFFF  # semantic label in lower half
        labels = self.remap_lut[labels]
        # crop a small set point clouds
        # pick_idx = np.random.choice(len(pc), 1)
        # center_point = pc[pick_idx, :].reshape(1, -1)

        select_idx = np.arange(pc.shape[0], dtype=np.int32)
        slt_idx = DP.shuffle_idx(select_idx)
        slt_pc = pc[select_idx]
        sel_lab = labels[select_idx]
        sel_remis = remis[select_idx]

        # augmentation
        if self.mode == 'training':
            if np.random.random() > 0.5:  # self.noisy_rot
                slt_pc = self.augment_noisy_rot(slt_pc, noisy_rot=self.noisy_rot)
            if np.random.random() > 0.5:  # self.noisy_rot
                slt_pc = self.augment_rot_z(slt_pc, rot_z=self.rot_z)

        sp_coords, sp_feats, sp_lab, unique_map, inverse_map = ME.utils.sparse_quantize(
                                                                                        coordinates=slt_pc,
                                                                                        features=slt_pc,
                                                                                        labels=sel_lab,
                                                                                        quantization_size=self.quantization_size,
                                                                                        return_index=True,
                                                                                        return_inverse=True)
        
        sel_remis = sel_remis[unique_map]
        sp_lab[sp_lab == -100] = 0
        
        sp_data = (sp_coords, sp_feats, sp_lab, inverse_map, unique_map)
        cloud_ind = np.array([cloud_ind], dtype=np.int32)

        return slt_pc, sel_lab, slt_pc, slt_idx, cloud_ind, sp_data

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
        # for sparse data
        coords, feats, labels, inverse_map, unique_map = list(zip(*sp_data))
        # Generate batched coordinates
        inputs['coords_mink'] = ME.utils.batched_coordinates(coords)
        # Concatenate all lists
        inputs['feats_mink'] = torch.from_numpy(np.concatenate(feats, 0)).float()
        inputs['labels_mink'] = torch.from_numpy(np.concatenate(labels, 0)).long()

        list_inverse_map = list(inverse_map)
        list_unique_map = list(unique_map)
        post_len, unique_len = 0, 0
        for i_list in range(len(list_inverse_map)):
            list_inverse_map[i_list] = list_inverse_map[i_list] + post_len
            post_len += unique_map[i_list].shape[0]

            list_unique_map[i_list] = list_unique_map[i_list] + unique_len
            unique_len += inverse_map[i_list].shape[0]

        inputs['inverse_map'] = torch.from_numpy(np.concatenate(list_inverse_map, 0)).long()
        inputs['unique_map'] = torch.from_numpy(np.concatenate(list_unique_map, 0)).long()

        return inputs
