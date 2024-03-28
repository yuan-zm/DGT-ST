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


class SynLiDAR_infer_B(SemanticDataset):
    def __init__(self, cfg, mode, downSample=False, data_list=None, stride=1):
        '''
        pass
        '''
        self.cfg = cfg
        self.name = 'SynLidar_Test'
        self.d_domain = 'source_infer_B'
        self.shift_prob = cfg.MODEL_G.aug_shift_prob

        self.dataset_path = os.path.expanduser(cfg.DATASET_SOURCE.DATASET_DIR)

        self.downSample = downSample

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

        # for sparse
        self.quantization_size = cfg.DATASET_SOURCE.VOXEL_SIZE

        self.mode = mode
        if data_list is None:
            if mode == 'gen_pselab':
                seq_list = ['00', '01', '02', '03', '04', '05', '06',
                            '07', '08', '09', '10', '11', '12']
            elif mode == 'test':
                seq_list = ['03']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)
        
        self.data_list = self.data_list[::stride]

        print('This is ** {} ** dataset, filepath ** {} ** mode is ** {} **, has ** {} ** scans.'.
                        format(self.name, self.dataset_path, self.mode, len(self.data_list)))

    # def __len__(self):
    #     return len(self.data_list)

    def __getitem__(self, item):
        # s_pc ==> selected_pc  p_lab ==> proj_labels
        slt_pc, s_lab, s_idx, cloud_ind, sp_data, sp_data_4, mask, slt_ps_lab = self.spatially_regular_gen(
            item, self.data_list)

        return slt_pc, s_lab, s_idx, cloud_ind, sp_data, sp_data_4, mask, slt_ps_lab, 0

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]

        pc, remis, labels = get_sk_data(pc_path, self.dataset_path, self.remap_lut, self.name)
        
        if self.downSample: #  and np.random.random() > 0.5
            if self.cfg.DATASET_TARGET.TYPE == 'SemanticPOSS':
                downsample_path = os.path.expanduser('~/dataset/SynLidar_DownSampled_Data/dataDownSample_Index/To_POSS')
            if self.cfg.DATASET_TARGET.TYPE == 'nuScenes':
                downsample_path = os.path.expanduser('~/dataset/SynLidar_DownSampled_Data/dataDownSample_Index/To_nuScenes')
            
            seq_id, frame_id = pc_path[0], pc_path[1]
            downsample_inds_path = join(downsample_path, seq_id, frame_id + '.npy')
            downsample_inds = np.load(downsample_inds_path)
        
            pc = pc[downsample_inds]
            labels = labels[downsample_inds]
            remis = remis[downsample_inds]

        select_idx = np.arange(pc.shape[0], dtype=np.int32)
        
        feats = np.concatenate((pc, remis.reshape(-1, 1)), axis=1)
      
        slt_ps_lab = np.zeros_like(pc)
        mask = np.zeros_like(labels).astype(np.float32)
        if self.cfg.DATASET_SOURCE.USE_INTENSITY and self.cfg.DATASET_TARGET.USE_INTENSITY:
            feats = np.concatenate((pc, remis.reshape(-1, 1)), axis=1)
      
        sp_coords, sp_feats, sp_lab, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=pc,
            features=feats if self.cfg.DATASET_SOURCE.USE_INTENSITY else pc,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0,
            return_index=True,
            return_inverse=True)
        sp_remis = remis[unique_map]

        
        cloud_ind = np.array([cloud_ind], dtype=np.int32)
        sp_data = (sp_coords, sp_feats, sp_lab,  inverse_map, unique_map, sp_remis, len(pc))

        v_coords_4, v_ft_4, v_lab_4, uni_map_4, inv_map_4 = ME.utils.sparse_quantize(
                                                                    coordinates=pc,
                                                                    features=feats if self.cfg.DATASET_SOURCE.USE_INTENSITY else pc,
                                                                    labels=labels,
                                                                    quantization_size=self.quantization_size * 4,
                                                                    return_index=True,
                                                                    return_inverse=True)
        sp_remis_4 = remis[uni_map_4]

        sp_data_4 = (v_coords_4, v_ft_4, v_lab_4, inv_map_4, uni_map_4, sp_remis_4)

        if self.downSample: # 这个是用来计算downsample后还有多少点的
            aug_sp_data = self.polar_range_drop(pc[unique_map], sp_lab, sp_remis)
            aug_sp_data = (aug_sp_data[0], aug_sp_data[1], aug_sp_data[2],  aug_sp_data[3], aug_sp_data[4], aug_sp_data[5], len(pc))
            return pc, labels, select_idx, cloud_ind, aug_sp_data, sp_data_4, mask, slt_ps_lab

        return pc, labels, select_idx, cloud_ind, sp_data, sp_data_4, mask, slt_ps_lab

    