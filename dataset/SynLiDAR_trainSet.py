from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import yaml
import os
from .dataset_base import SemanticDataset


class SynLiDAR_Dataset(SemanticDataset):
    def __init__(self, cfg, mode, data_list=None):
        self.cfg = cfg
        self.name = 'SynLiDAR'
        self.d_domain = 'source'
        
        if self.cfg.DATASET_SOURCE.USE_DGT:
            self.total_beams = cfg.DATASET_SOURCE.total_beams
        
        self.shift_prob = cfg.MODEL_G.aug_shift_prob
        self.shift_range = cfg.MODEL_G.aug_shift_range
        self.aug_data_prob = cfg.MODEL_G.aug_data_prob

        self.dataset_path = os.path.expanduser(cfg.DATASET_SOURCE.DATASET_DIR)

        self.in_num_voxels = cfg.DATASET_SOURCE.IN_NUM_VOXELS

        DATA = yaml.safe_load(open('utils/annotations.yaml', 'r'))
        if cfg.DATASET_TARGET.TYPE == 'SemanticKITTI':
            remap_dict = DATA["map_2_semantickitti"]
            max_key = max(remap_dict.keys())
            self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

            self.label_name = DATA["map_2_semantickitti_labels"]
        
        elif cfg.DATASET_TARGET.TYPE == 'SemanticPOSS':
            remap_dict = DATA["map_2_semanticposs"]
            max_key = max(remap_dict.keys())
            self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

            self.label_name = DATA["map_2_semanticposs_labels"]
        
        elif cfg.DATASET_TARGET.TYPE == 'nuScenes':
            DATA = yaml.safe_load(open('utils/annotations_Syn2nuScenes_AdversarialMask.yaml', 'r'))
            remap_dict = DATA["map_2_nuScenes_AMask"]
            max_key = max(remap_dict.keys())
            self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

            self.label_name = DATA["map_2_nuscenes_AMask_labels_name"]

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
                seq_list = ['00', '01', '02', '03', '04', '05', '06',
                            '07', '08', '09', '10', '11', '12']
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

    