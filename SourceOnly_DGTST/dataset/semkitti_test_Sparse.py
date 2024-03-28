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
from .data_utils import pc_normalize, get_sk_data
# slt --> selected


class SemanticKITTI(SemanticBase):
    def __init__(self, cfg, mode, data_list=None):
      
        self.name = 'SemanticKITTI_Test'
        self.dataset_path = os.path.expanduser('~/dataset/semanticKITTI/dataset/sequences')

        DATA = yaml.safe_load(open('utils/semantic-kitti.yaml', 'r'))
        remap_dict = DATA["learning_map"]
        max_key = max(remap_dict.keys())
        self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        # self.learning_map_inv = DATA["learning_map_inv"]
        self.label_name = DATA["labels"]

        self.num_classes = cfg.MODEL_G.NUM_CLASSES
        self.ignored_labels = np.sort([0])

        # for sparse
        self.quantization_size = 0.05

        self.mode = mode
        if data_list is None:
            if mode == 'gen_pselab':
                seq_list = ['00', '01', '02', '03',
                            '04', '05', '06', '07', '09', '10']
            elif mode == 'test':
                seq_list = ['08']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)
       
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # s_pc ==> selected_pc  p_lab ==> proj_labels
        slt_pc, s_lab, s_idx, cloud_ind, sp_data = self.spatially_regular_gen(
            item, self.data_list)

        return slt_pc, s_lab, s_idx, cloud_ind, sp_data

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]

        pc, remis, tree, labels = self.get_data(self.dataset_path, pc_path)  # get data
        
        labels = labels & 0xFFFF  # semantic label in lower half
        labels = self.remap_lut[labels]

        # slt_idx = np.arange(len(pc))
        select_idx = np.arange(pc.shape[0], dtype=np.int32)
        # select_idx = DP.shuffle_idx(select_idx)
        # pc = pc[select_idx]
        # labels = labels[select_idx]
        # remis = remis[select_idx]
        
        sp_coords, sp_feats, sp_lab, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=pc,
            features=pc,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0,
            return_index=True,
            return_inverse=True)

        cloud_ind = np.array([cloud_ind], dtype=np.int32)
        sp_data = (sp_coords, sp_feats, sp_lab, inverse_map, unique_map, len(pc))
        # pc = self.pc_z_norm(pc)
        return pc, labels, select_idx, cloud_ind, sp_data
