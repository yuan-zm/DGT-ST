from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import yaml
import os
from .dataset_base import SemanticDataset


class semPoss_Dataset(SemanticDataset):
    def __init__(self, cfg, mode, data_list=None):
        self.cfg = cfg
        self.name = 'SemanticPOSS'
        self.dataset_path = os.path.expanduser(cfg.DATASET_TARGET.DATASET_DIR)
        self.d_domain = 'target'

        self.shift_prob = cfg.MODEL_G.aug_shift_prob
        self.shift_range = cfg.MODEL_G.aug_shift_range
        self.aug_data_prob = cfg.MODEL_G.aug_data_prob

        DATA = yaml.safe_load(open('utils/semantic-poss.yaml', 'r'))
        remap_dict = DATA["learning_map"]
        max_key = max(remap_dict.keys())
        self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        self.label_name = DATA["labels"]

        self.num_classes = cfg.MODEL_G.NUM_CLASSES
        self.ignored_labels = np.sort([0])

        # for augmentation
        self.noisy_rot = 0.1
        self.flip_y = 0.5
        self.rot_z = 6.2831  # 2 * pi

        # for sparse
        self.quantization_size = cfg.DATASET_TARGET.VOXEL_SIZE
        self.in_num_voxels = cfg.DATASET_TARGET.IN_NUM_VOXELS

        self.mode = mode
        if data_list is None:
            if mode == 'training':
                seq_list = ['00', '01', '02', '05', '04']
            elif mode == 'validation':
                seq_list = ['03']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list

        self.data_list = sorted(self.data_list)
        self.data_list_ori = sorted(self.data_list).copy()

        print('This is ** {} ** dataset, filepath ** {} ** mode is ** {} **, has ** {} ** scans.'.
                        format(self.name, self.dataset_path, self.mode, len(self.data_list)))


    def __getitem__(self, item):
        slt_pc, s_lab, s_idx, cloud_ind, sp_data, raw_sp_data = self.gen_sample(item)

        return slt_pc, s_lab, s_idx, cloud_ind, sp_data, raw_sp_data

    