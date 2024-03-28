from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import yaml
import os
import copy

from .data_utils import get_sk_data, augment_noisy_rot, augment_rot_z
import MinkowskiEngine as ME
from dataset.downsample_utils import get_specific_beam_mask


class SemanticDataset(torch_data.Dataset):
    def __init__(self, mode, data_list=None):
        pass

    def get_class_weight(self):
        return DP.get_class_weights(self.dataset_path, self.data_list_ori, self.num_classes, self.remap_lut)

    def __len__(self):
        if self.cfg.TRAIN.DEBUG:
            return 20
        else:
            return len(self.data_list)
            
    def __getitem__(self, item):
        return None

    def gen_sample(self, cloud_ind):  # Generator loop

        pc_name = self.data_list[cloud_ind]
        # get one frame sample
        slt_pc, slt_remis, slt_lab = get_sk_data(pc_name, self.dataset_path, self.remap_lut, self.name)
        
        if self.cfg.DATASET_SOURCE.USE_DGT and \
            self.cfg.DATASET_TARGET.TYPE != 'SemanticKITTI' and \
                self.name == 'SynLiDAR' and \
                    self.mode == 'training': #  and np.random.random() > 0.5
            
            beamLabel_path = os.path.expanduser('change_data/SynLiDAR_beam_label')
            seq_id, frame_id = pc_name[0], pc_name[1]
            beamLabel_inds_path = join(beamLabel_path, seq_id, frame_id + '.npy')
            beamLabel = np.load(beamLabel_inds_path)
            
            total_beam_labels = np.arange(0, self.total_beams)

            if self.cfg.DATASET_TARGET.TYPE == 'SemanticPOSS': # 40 beams
                choose_beams = total_beam_labels[::2] # chose 32 beams first
                rest_choose_beams = total_beam_labels[1::2][:8] # chose another 8 beams near the LiDAR center
                choose_beams = np.concatenate((choose_beams, rest_choose_beams), axis=0)
            if self.cfg.DATASET_TARGET.TYPE == 'nuScenes': # 32 beams
                choose_beams = total_beam_labels[::2] # only use 32 beams

            choseInds = get_specific_beam_mask(beamLabel, choose_beams)
            slt_pc = slt_pc[choseInds]
            slt_lab = slt_lab[choseInds]
            slt_remis = slt_remis[choseInds]

        # augmentation
        if self.mode == 'training':
            if np.random.random() > 0.5:
                slt_pc = augment_noisy_rot(slt_pc, noisy_rot=self.noisy_rot)
            if np.random.random() > 0.5:
                slt_pc = augment_rot_z(slt_pc, rot_z=self.rot_z)
        # construct a sparse Tensor
        if self.cfg.DATASET_SOURCE.USE_INTENSITY and self.cfg.DATASET_TARGET.USE_INTENSITY:
            feature = np.concatenate((slt_pc, slt_remis.reshape(-1, 1)), axis=1)
        v_coords, v_ft, v_lab, uni_map, inv_map = ME.utils.sparse_quantize(
                                                    coordinates=slt_pc,
                                                    features=feature if self.cfg.DATASET_SOURCE.USE_INTENSITY else slt_pc,
                                                    labels=slt_lab,
                                                    quantization_size=self.quantization_size,
                                                    return_index=True,
                                                    return_inverse=True)
        v_remis = slt_remis[uni_map]

        
        # choose fix input num of voxels
        slt_idx = None
        if self.mode == 'training' and self.in_num_voxels > 0:
            if len(v_coords) > self.in_num_voxels:
                slt_idx = np.random.choice(len(v_coords), self.in_num_voxels, replace=False)
                v_coords = v_coords[slt_idx]
                v_ft = v_ft[slt_idx]
                v_lab = v_lab[slt_idx]
                v_remis = v_remis[slt_idx]

        v_lab[v_lab == -100] = 0
        cloud_ind = np.array([cloud_ind], dtype=np.int32)

        
        if self.d_domain == 'target':  # 保险点 让slt_lab置为0
            slt_lab = np.zeros_like(slt_lab)
            v_lab = np.zeros_like(v_lab)

        sp_data = (v_coords, v_ft, v_lab, inv_map, uni_map, v_remis)

        if self.cfg.DATASET_SOURCE.USE_DGT and np.random.random() > self.aug_data_prob:
            aug_sp_data = self.dgt(v_ft, v_lab, v_remis)
        else:
            del_mask = np.ones_like(v_lab)
            aug_sp_data = (v_coords, v_ft, v_lab, inv_map, uni_map, v_remis, del_mask)
       
        return slt_pc, slt_lab, slt_idx, cloud_ind, sp_data, aug_sp_data

    def dgt(self, pc, lab, remis):
        # Initially, I thought about discarding half of the scan to ensure the discriminability of the segmentation model.
        # However, my experimental results show that this setting does not have much impact on the final result.
        start_angle = (np.random.random() - 1) * np.pi
        end_angle = start_angle + np.pi

        yaw = -np.arctan2(pc[:, 1], pc[:, 0])

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
                this_range_ind = np.where((xy_dis > dis_range[i]) & (xy_dis < dis_range[i+1]))[0] #  & (yaw > start_angle) & (yaw < end_angle)
                this_del_num = int(len(this_range_ind) * drop_prob[i])
                this_drop_ind = np.random.choice(this_range_ind, this_del_num, replace=False)
                del_inds.extend(this_drop_ind)
        
        aug_drop_pc = np.delete(copy.deepcopy(pc), del_inds, axis=0)
        aug_drop_remis = np.delete(copy.deepcopy(remis), del_inds, axis=0)
        aug_drop_lab = np.delete(copy.deepcopy(lab), del_inds, axis=0)
        del_mask[del_inds] = 0

        # add XY shift
        if np.random.random() > self.shift_prob and self.d_domain == 'source':
            N, C = aug_drop_pc.shape
            shift_range = self.shift_range 
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

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def collate_fn(self, batch):
        sp_data = []
        slt_pc, slt_lab, slt_idx, cloud_ind = [], [], [], []
        aug_sp_data = []
        for i in range(len(batch)):
            slt_pc.append(batch[i][0])
            slt_lab.append(batch[i][1])
            slt_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])
            sp_data.append(batch[i][4])
            if not (self.d_domain == 'source_infer_B' or self.d_domain == 'target_infer_B'):
                aug_sp_data.append(batch[i][5])
        # slt_pc = np.stack(slt_pc)
        # slt_lab = np.stack(slt_lab)
        # slt_idx = np.stack(slt_idx)
        cloud_ind = np.stack(cloud_ind)

        inputs = {}
        inputs['pc_labs'] = torch.from_numpy(np.concatenate(slt_lab)).long()
        inputs['cloud_inds'] = torch.from_numpy(cloud_ind).long()
        # get sparse data
        if self.d_domain == 'source_infer_B' or self.d_domain == 'target_infer_B':
            coords, feats, labels, inverse_map, unique_map, sp_remis, s_len = list(zip(*sp_data))
            inputs['s_lens'] = torch.from_numpy(np.stack(s_len, 0)).long()
        elif self.d_domain == 'source_infer_B_aug' or self.d_domain == 'target_infer_B_aug':
            coords, feats, labels, inverse_map, unique_map, sp_remis = list(zip(*sp_data))
            aug_coords, aug_feats, aug_labels, aug_inverse_map, aug_unique_map, aug_sp_remis, del_mask = list(zip(*aug_sp_data))
            # count_len_bi = 0
            # list_del_ind = list(del_ind)
            # for bi in range(len(inverse_map)):
            #     list_del_ind[bi] = list_del_ind[bi] + count_len_bi
            #     count_len_bi += len(aug_inverse_map[bi])
            inputs['aug_del_mask'] = torch.from_numpy(np.concatenate(del_mask, 0)).bool()
            inputs['aug_sp_remis'] =  torch.from_numpy(np.concatenate(aug_sp_remis, 0)).float()
    
            inputs['aug_coords_mink'] = ME.utils.batched_coordinates(aug_coords)
            inputs['aug_feats_mink'] = torch.from_numpy(np.concatenate(aug_feats, 0)).float()
            inputs['aug_labels_mink'] = torch.from_numpy(np.concatenate(aug_labels, 0)).long()

            inputs['aug_inverse_map'], inputs['aug_unique_map'] = self.get_inv_unq_map(aug_inverse_map, aug_unique_map) 

        else:
            coords, feats, labels, inverse_map, unique_map, sp_remis = list(zip(*sp_data))
            if self.mode == 'training':
                aug_coords, aug_feats, aug_labels, aug_inverse_map, aug_unique_map, aug_sp_remis, del_mask = list(zip(*aug_sp_data))
               
                inputs['aug_del_mask'] = torch.from_numpy(np.concatenate(del_mask, 0)).bool()
                inputs['aug_sp_remis'] =  torch.from_numpy(np.concatenate(aug_sp_remis, 0)).float()
        
                inputs['aug_coords_mink'] = ME.utils.batched_coordinates(aug_coords)
                inputs['aug_feats_mink'] = torch.from_numpy(np.concatenate(aug_feats, 0)).float()
                inputs['aug_labels_mink'] = torch.from_numpy(np.concatenate(aug_labels, 0)).long()

                inputs['aug_inverse_map'], inputs['aug_unique_map'] = self.get_inv_unq_map(aug_inverse_map, aug_unique_map) 

        inputs['sp_remis'] =  torch.from_numpy(np.concatenate(sp_remis, 0)).float()
     
        # Generate batched coordinates
        inputs['coords_mink'] = ME.utils.batched_coordinates(coords)
        # Concatenate all lists
        inputs['feats_mink'] = torch.from_numpy(np.concatenate(feats, 0)).float()
        inputs['labels_mink'] = torch.from_numpy(np.concatenate(labels, 0)).long()

        inputs['inverse_map'], inputs['unique_map'] = self.get_inv_unq_map(inverse_map, unique_map) 

        return inputs
    
    def get_inv_unq_map(self, inverse_map, unique_map):
        # 如果单纯的将一个batch的inverse——map合并起来，那么都会在0-40960之间所以出错
        list_inverse_map = list(inverse_map)
        list_unique_map = list(unique_map)
        post_len, unique_len = 0, 0

        for i_list in range(len(list_inverse_map)):
            list_inverse_map[i_list] = list_inverse_map[i_list] + post_len
            post_len += unique_map[i_list].shape[0]

            list_unique_map[i_list] = list_unique_map[i_list] + unique_len
            unique_len += inverse_map[i_list].shape[0]
        
        return torch.from_numpy(np.concatenate(list_inverse_map, 0)).long(), torch.from_numpy(np.concatenate(list_unique_map, 0)).long()
        