from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import yaml
import os
from .dataset_base import SemanticDataset
import MinkowskiEngine as ME
from .data_utils import pc_normalize, get_sk_data
# slt --> selected


class SemanticKITTI_infer(SemanticDataset):
    def __init__(self, cfg, mode, d_domian='target_test', source_dataset=None,
                 data_list=None, once_infer=False, infer_bs=1, stride=1):
        '''
        once_infer=False, infer_bs=0 only used for fast inference  
        '''
        self.cfg = cfg
        self.name = 'SemanticKITTI'

        self.dataset_path = os.path.expanduser(cfg.DATASET_TARGET.DATASET_DIR)

        self.d_domain = 'target_infer'

        DATA = yaml.safe_load(open('utils/semantic-kitti.yaml', 'r'))
        remap_dict = DATA["learning_map"]
        max_key = max(remap_dict.keys())
        self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        self.label_name = DATA["labels"]

        self.num_classes = cfg.MODEL_G.NUM_CLASSES
        self.ignored_labels = np.sort([0])

        # for sparse
        self.quantization_size = cfg.DATASET_TARGET.VOXEL_SIZE

        self.mode = mode
        if data_list is None:
            if mode == 'gen_pselab':
                seq_list = ['00', '01', '02', '03', '04',
                            '05', '06', '07', '09', '10']
            elif mode == 'test':
                seq_list = ['08']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)
        if once_infer:
            self.data_list = sorted(self.data_list * infer_bs)
        self.data_list = self.data_list[::stride]

        print('This is ** {} ** dataset, filepath ** {} ** mode is ** {} **, has ** {} ** scans.'.
                        format(self.name, self.dataset_path, self.mode, len(self.data_list)))

    def __getitem__(self, item):
        # s_pc ==> selected_pc  p_lab ==> proj_labels
        slt_pc, s_lab, s_idx, cloud_ind, sp_data, sp_data_4 = self.gen_sample(item)

        return slt_pc, s_lab, s_idx, cloud_ind, sp_data, sp_data_4

    def gen_sample(self, item):
        # Generator loop
        cloud_ind = item
        pc_path = self.data_list[cloud_ind]
        pc, remis, labels = get_sk_data(pc_path, self.dataset_path,
                                        self.remap_lut, self.name)

        # slt_idx = np.arange(len(pc))
        select_idx = np.arange(pc.shape[0], dtype=np.int32)
        # select_idx = DP.shuffle_idx(select_idx)
        # pc = pc[select_idx]
        # labels = labels[select_idx]
        # remis = remis[select_idx]

        # feats = np.concatenate((pc, remis.reshape(-1, 1)), axis=1)
        feats = pc
        sp_coords, sp_feats, sp_lab, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=pc,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0,
            return_index=True,
            return_inverse=True)

        remis_sp = remis[unique_map]
        
        v_coords_4, v_ft_4, v_lab_4, uni_map_4, inv_map_4 = ME.utils.sparse_quantize(
                                                                    coordinates=pc,
                                                                    features=feats ,
                                                                    labels=labels,
                                                                    quantization_size=self.quantization_size * 4,
                                                                    return_index=True,
                                                                    return_inverse=True)
        v_remis_4 = remis[uni_map_4]

        cloud_ind = np.array([cloud_ind], dtype=np.int32)
        sp_data = (sp_coords, sp_feats, sp_lab, inverse_map, unique_map, remis_sp)
        sp_data_4 = (v_coords_4, v_ft_4, v_lab_4, inv_map_4, uni_map_4, v_remis_4)

        return pc, labels, select_idx, cloud_ind, sp_data, sp_data_4
