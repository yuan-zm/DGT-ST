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

from dataset.downsample_utils import compute_angles, beam_label, generate_mask, generate_choosed_mask, change_beam_label

class SynLiDAR_infer_B(SemanticDataset):
    def __init__(self, cfg, mode, data_list=None, get_beam_label=False, stride=1):
        '''
        pass
        '''
        self.cfg = cfg
        self.name = 'SynLidar_Test'
        self.d_domain = 'source_infer_B'

        self.get_beam_label = get_beam_label

        self.dataset_path = os.path.expanduser(cfg.DATASET_SOURCE.DATASET_DIR)

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
        slt_pc, s_lab, s_idx, cloud_ind, sp_data, sp_data_4, mask, slt_ps_lab = self.spatially_regular_gen(
            item, self.data_list)

        return slt_pc, s_lab, s_idx, cloud_ind, sp_data, sp_data_4, mask, slt_ps_lab, 0

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]

        pc, remis, labels = get_sk_data(pc_path, self.dataset_path, self.remap_lut, self.name)

        if self.get_beam_label:
            beam = 64
            pc_np = pc[:, :3]

            theta, phi = compute_angles(pc_np)
            label, centroids = beam_label(theta, beam)
            idxs = np.argsort(centroids)

            seq_id, frame_id = pc_path[0], pc_path[1]

            new_beam_label = change_beam_label(label, idxs)
            # 保存 线 label
            save_base_path = "change_data/Syn_beam_label"
            mask_base_path = join(save_base_path, seq_id)
            os.makedirs(mask_base_path, exist_ok=True)
            beam_label_save_path = join(mask_base_path, frame_id + '.npy')
            
            np.save(beam_label_save_path, new_beam_label)

        # beam = 64
        # pc_np = pc[:, :3]

        # theta, phi = compute_angles(pc_np)
        # label, centroids = beam_label(theta, beam)
        # idxs = np.argsort(centroids)

        # seq_id, frame_id = pc_path[0], pc_path[1]
        # # for semanticPOSS
        # choose_idxs = idxs[::2] # 先拿32
        # rest_choose_idxs = idxs[1::2][:8] # 再拿 8个
        # choose_idxs = np.concatenate((choose_idxs, rest_choose_idxs), axis=0)

        # mask = generate_choosed_mask(phi, beam, label, choose_idxs, beam_ratio=2, bin_ratio=1)
        # save_ind_POSS = np.where(mask==True)[0]

        # save_base_path = "change_data/To_POSS"
        # mask_base_path = join(save_base_path, seq_id)
        # os.makedirs(mask_base_path, exist_ok=True)

        # POSS_save_path = join(mask_base_path, frame_id + '.npy')
        # np.save(POSS_save_path, save_ind_POSS)
        
        # # for nuScenes
        # nuScenes_choose_idxs = idxs[::2] # 拿32
        # mask_nuScenes = generate_choosed_mask(phi, beam, label, nuScenes_choose_idxs, beam_ratio=2, bin_ratio=1)
        # save_ind_nuScenes = np.where(mask_nuScenes==True)[0]

        # save_base_path = "change_data/To_nuScenes"
        # mask_base_path = join(save_base_path, seq_id)
        # os.makedirs(mask_base_path, exist_ok=True)
        # nuScenes_save_path = join(mask_base_path, frame_id + '.npy')
        # np.save(nuScenes_save_path, save_ind_nuScenes)

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

        return pc, labels, select_idx, cloud_ind, sp_data, sp_data_4, mask, slt_ps_lab
