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
from .data_utils import pc_normalize, get_sk_data, get_sk_DownSampled_data
# slt --> selected
import copy

class SynLiDAR_infer_B_To_POSS(SemanticDataset):
    def __init__(self, cfg, mode, data_list=None, stride=1):
        '''
        pass
        '''
        self.cfg = cfg
        self.name = 'SynLidar_Test'
        self.d_domain = 'source_infer_B_aug'

        self.dataset_path = os.path.expanduser(cfg.DATASET_SOURCE.DATASET_DIR)

        self.shift_prob = cfg.MODEL_G.aug_shift_prob
        self.shift_range = cfg.MODEL_G.aug_shift_range

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
            DATA = yaml.safe_load(open('utils/annotations_Syn2nuScenes.yaml', 'r'))
            remap_dict = DATA["map_2_nuScenes"]
            max_key = max(remap_dict.keys())
            self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

            self.label_name = DATA["map_2_nuscenes_labels_name"]

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
        slt_pc, s_lab, s_idx, cloud_ind, sp_data, aug_sp_data = self.spatially_regular_gen(
            item, self.data_list)

        return slt_pc, s_lab, s_idx, cloud_ind, sp_data, aug_sp_data

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]

        pc, remis, labels = get_sk_data(pc_path, self.dataset_path, self.remap_lut, self.name)
        
        # if self.cfg.DATASET_TARGET.TYPE == 'SemanticPOSS':
        #     downsample_path = os.path.expanduser('~/dataset/SynLidar_DownSampled_Data/dataDownSample_Index/To_POSS')
        # if self.cfg.DATASET_TARGET.TYPE == 'nuScenes':
        #     downsample_path = os.path.expanduser('~/dataset/SynLidar_DownSampled_Data/dataDownSample_Index/To_nuScenes')
        
        # seq_id, frame_id = pc_path[0], pc_path[1]
        # downsample_inds_path = join(downsample_path, seq_id, frame_id + '.npy')
        # downsample_inds = np.load(downsample_inds_path)
    
        # pc = pc[downsample_inds]
        # labels = labels[downsample_inds]
        # remis = remis[downsample_inds]
        
        select_idx = np.arange(pc.shape[0], dtype=np.int32)
              
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
        sp_data = (sp_coords, sp_feats, sp_lab,  inverse_map, unique_map, sp_remis)

        # v_coords_4, v_ft_4, v_lab_4, uni_map_4, inv_map_4 = ME.utils.sparse_quantize(
        #                                                             coordinates=pc,
        #                                                             features=feats if self.cfg.DATASET_SOURCE.USE_INTENSITY else pc,
        #                                                             labels=labels,
        #                                                             quantization_size=self.quantization_size * 4,
        #                                                             return_index=True,
        #                                                             return_inverse=True)
        # sp_remis_4 = remis[uni_map_4]

        # sp_data_4 = (v_coords_4, v_ft_4, v_lab_4, inv_map_4, uni_map_4, sp_remis_4)
        
        aug_sp_data = self.polar_range_drop(sp_feats, sp_lab, sp_remis)

        return pc, labels, select_idx, cloud_ind, sp_data, aug_sp_data

    def polar_range_drop(self, pc, lab, remis):
        start_angle = (np.random.random() - 1) * np.pi
        end_angle = start_angle + np.pi

        yaw = -np.arctan2(pc[:, 1], pc[:, 0])
        chose_idx = np.where((yaw > start_angle) & (yaw < end_angle))

        # Source density-aware augmentation 
        del_inds = []
        del_mask = np.ones_like(lab)

        if self.name == 'SynLiDAR' and self.mode == 'training': #  and np.random.random() > 0.5
            src2tgt_denseity_raio = np.array(self.cfg.DATASET_TARGET.DENSITY) / (np.array(self.cfg.DATASET_SOURCE.DENSITY) + 1e-10)
            drop_prob = 1 - np.clip(src2tgt_denseity_raio, a_min=0, a_max=1.)
        else:
            tgt2src_denseity_raio = np.array(self.cfg.DATASET_SOURCE.DENSITY) / (np.array(self.cfg.DATASET_TARGET.DENSITY) + 1e-10)
            drop_prob = 1 - np.clip(tgt2src_denseity_raio, a_min=0, a_max=1.)

        xy_dis = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        dis_range = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(np.float) # np.linspace(0, 100, 10)
        # dis_range = np.linspace(0, 100, 10)
        for i in range(10):
            if drop_prob[i] > 0:
                this_range_ind = np.where((xy_dis > dis_range[i]) & (xy_dis < dis_range[i+1]))[0]  # a  & (yaw > start_angle) & (yaw < end_angle)
                this_del_num = int(len(this_range_ind) * drop_prob[i])
                this_drop_ind = np.random.choice(this_range_ind, this_del_num, replace=False)
                del_inds.extend(this_drop_ind)
        
        aug_drop_pc = np.delete(copy.deepcopy(pc), del_inds, axis=0)
        aug_drop_remis = np.delete(copy.deepcopy(remis), del_inds, axis=0)
        aug_drop_lab = np.delete(copy.deepcopy(lab), del_inds, axis=0)
        del_mask[del_inds] = 0

        # add shift
        if np.random.random() > self.shift_prob:
            N, C = aug_drop_pc.shape
            shift_range = self.shift_range #  0.1 # 05
            assert(shift_range > 0)
            shifts = np.random.uniform(-shift_range, shift_range, (N, C))

            end_shifts = np.zeros_like(shifts)
            shift_chose_ind = np.random.choice(shifts.shape[0], int(shifts.shape[0]* 0.1), replace=False)
            end_shifts[shift_chose_ind] = shifts[shift_chose_ind]
            aug_drop_pc[:, :2] += end_shifts[:, :2]

        aug_v_coords, aug_v_ft, aug_v_lab, aug_uni_map, aug_inv_map = ME.utils.sparse_quantize(
                                                                                                coordinates=aug_drop_pc,
                                                                                                features=aug_drop_pc,
                                                                                                labels=aug_drop_lab,
                                                                                                quantization_size=self.quantization_size,
                                                                                                return_index=True,
                                                                                                return_inverse=True)
        aug_v_remis = aug_drop_remis[aug_uni_map]

        aug_v_lab[aug_v_lab == -100] = 0

        aug_sp_data = (aug_v_coords, aug_v_ft, aug_v_lab, aug_inv_map, aug_uni_map, aug_v_remis, del_mask)
        return aug_sp_data
