import numpy as np

import torch    
import torch.nn.functional as F
import MinkowskiEngine as ME

def laserMix(cfg, src_BData, tgt_BData):
    
    # 一个batch的source data
    if cfg.DATASET_SOURCE.use_aug_for_laserMix:
        batch_source_pts = src_BData['aug_coords_mink']
        batch_source_labels = src_BData['aug_labels_mink']
        batch_source_features = src_BData['aug_feats_mink']
    else:
        batch_source_pts = src_BData['coords_mink']
        batch_source_labels = src_BData['labels_mink']
        batch_source_features = src_BData['feats_mink']

    if cfg.DATASET_TARGET.use_aug_for_laserMix:
        batch_target_idx = tgt_BData['aug_coords_mink'][:, 0]
        batch_target_pts = tgt_BData['aug_coords_mink']
        batch_target_features = tgt_BData['aug_feats_mink']
        # 得到伪标签
        batch_target_labels = tgt_BData['aug_pseudo_label']
    else:
        batch_target_idx = tgt_BData['coords_mink'][:, 0]
        batch_target_pts = tgt_BData['coords_mink']
        batch_target_features = tgt_BData['feats_mink']
        # 得到伪标签
        batch_target_labels = tgt_BData['pseudo_label']

    # 得到batchsize
    batch_size = int(torch.max(batch_target_idx).item() + 1)
    
    new_batch = {'masked_target_pts': [],
                 'masked_target_labels': [],
                 'masked_target_features': [],
                 'masked_target_mask': [],
                 'masked_source_pts': [],
                 'masked_source_labels': [],
                 'masked_source_features': [],
                 'masked_source_mask': []}

    target_order = torch.arange(batch_size)
                                                                           
    for b in range(batch_size):
        source_b_idx = batch_source_pts[:, 0] == b
        target_b = target_order[b]
        target_b_idx = batch_target_idx == target_b

        # source
        source_pts = batch_source_pts[source_b_idx, 1:] * cfg.DATASET_SOURCE.VOXEL_SIZE
        source_labels = batch_source_labels[source_b_idx]
        source_features = batch_source_features[source_b_idx]

        # target
        target_pts = batch_target_pts[target_b_idx, 1:] * cfg.DATASET_SOURCE.VOXEL_SIZE
        target_labels = batch_target_labels[target_b_idx]
        target_features = batch_target_features[target_b_idx]

        pitch_angles = [-25, 3]
        num_areas = [3, 4, 5, 6]

        pitch_angle_down = pitch_angles[0] / 180 * np.pi
        pitch_angle_up = pitch_angles[1] / 180 * np.pi

        rho_src = torch.sqrt(source_pts[:, 0]**2 + source_pts[:, 1]**2)
        pitch_src = torch.atan2(source_pts[:, 2], rho_src)
        pitch_src = torch.clamp(pitch_src, pitch_angle_down + 1e-5, pitch_angle_up - 1e-5)

        rho_tgt = torch.sqrt(target_pts[:, 0]**2 + target_pts[:, 1]**2)
        pitch_tgt = torch.atan2(target_pts[:, 2], rho_tgt)
        pitch_tgt = torch.clamp(pitch_tgt, pitch_angle_down + 1e-5, pitch_angle_up - 1e-5)

        num_areas = np.random.choice(num_areas, size=1)[0]
        angle_list = np.linspace(pitch_angle_up, pitch_angle_down, num_areas + 1)
        
        mask_source = torch.ones(source_pts.shape[0]) # source mask =1 
        mask_target = torch.zeros(target_pts.shape[0]) # target mask =0 

        points_mix1_idx_src, points_mix1_idx_tgt = [], []
        points_mix2_idx_src, points_mix2_idx_tgt = [], []

        points_mix1, points_mix2 = [], []
        labels_mix1, labels_mix2 = [], []
        masks_mix1, masks_mix2 = [], []

        for i in range(num_areas):
            # convert angle to radian
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]
            idx_src = (pitch_src > start_angle) & (pitch_src <= end_angle)
            idx_tgt = (pitch_tgt > start_angle) & (pitch_tgt <= end_angle)
            if i % 2 == 0:  # pick from original point cloud
                points_mix1.append(source_pts[idx_src])
                labels_mix1.append(source_labels[idx_src])
                masks_mix1.append(mask_source[idx_src])
        
                points_mix2.append(target_pts[idx_tgt])
                labels_mix2.append(target_labels[idx_tgt])
                masks_mix2.append(mask_target[idx_tgt])

            else:  # pickle from mixed point cloud
                points_mix1.append(target_pts[idx_tgt])
                labels_mix1.append(target_labels[idx_tgt])
                masks_mix1.append(mask_target[idx_tgt])

                points_mix2.append(source_pts[idx_src])
                labels_mix2.append(source_labels[idx_src])
                masks_mix2.append(mask_source[idx_src])

        masked_target_pts = torch.cat(points_mix1)
        masked_target_labels = torch.cat(labels_mix1)
        masked_target_features = torch.cat(points_mix1) # torch.ones(masked_target_pts.shape[0]).view(-1, 1).cuda()
        masked_target_mask = torch.cat(masks_mix1)
                                                                                                   
        masked_source_pts = torch.cat(points_mix2)
        masked_source_labels = torch.cat(labels_mix2)
        masked_source_features = torch.cat(points_mix2) # torch.ones(masked_source_pts.shape[0]).view(-1, 1).cuda()
        masked_source_mask = torch.cat(masks_mix2)
                                                                                                   
        # # masked_target/source_mask 本域为0 另一个域为1
        # masked_target_pts, masked_target_labels, masked_target_features, masked_target_mask = mask(origin_pts=source_pts,
        #                                                                                             origin_labels=source_labels,
        #                                                                                             origin_features=source_features,
        #                                                                                             dest_pts=target_pts,
        #                                                                                             dest_labels=target_labels,
        #                                                                                             dest_features=target_features)

        # masked_source_pts, masked_source_labels, masked_source_features, masked_source_mask = mask(origin_pts=target_pts,
        #                                                                                             origin_labels=target_labels,
        #                                                                                             origin_features=target_features,
        #                                                                                             dest_pts=source_pts,
        #                                                                                             dest_labels=source_labels,
        #                                                                                             dest_features=source_features,
        #                                                                                             is_pseudo=True)

        _, _,  masked_target_voxel_idx = ME.utils.sparse_quantize(coordinates=masked_target_pts,
                                                                        features=masked_target_features,
                                                                        # labels=masked_target_labels,
                                                                        quantization_size=cfg.DATASET_SOURCE.VOXEL_SIZE,
                                                                        return_index=True)

        _, _,  masked_source_voxel_idx = ME.utils.sparse_quantize(coordinates=masked_source_pts,
                                                                    features=masked_source_features,
                                                                    # labels=masked_source_labels,
                                                                    quantization_size=cfg.DATASET_SOURCE.VOXEL_SIZE,
                                                                    return_index=True)

        masked_target_pts = masked_target_pts[masked_target_voxel_idx]
        masked_target_labels = masked_target_labels[masked_target_voxel_idx]
        masked_target_features = masked_target_features[masked_target_voxel_idx]
        masked_target_mask = masked_target_mask[masked_target_voxel_idx]

        masked_source_pts = masked_source_pts[masked_source_voxel_idx]
        masked_source_labels = masked_source_labels[masked_source_voxel_idx]
        masked_source_features = masked_source_features[masked_source_voxel_idx]
        masked_source_mask = masked_source_mask[masked_source_voxel_idx]

        masked_target_pts = torch.floor(masked_target_pts / cfg.DATASET_SOURCE.VOXEL_SIZE)
        masked_source_pts = torch.floor(masked_source_pts / cfg.DATASET_SOURCE.VOXEL_SIZE)

        batch_index = torch.ones([masked_target_pts.shape[0], 1]).cuda() * b
        masked_target_pts = torch.cat([batch_index, masked_target_pts], dim=-1)

        batch_index = torch.ones([masked_source_pts.shape[0], 1]).cuda() * b
        masked_source_pts = torch.cat([batch_index, masked_source_pts], dim=-1)

        new_batch['masked_target_pts'].append(masked_target_pts)
        new_batch['masked_target_labels'].append(masked_target_labels)
        new_batch['masked_target_features'].append(masked_target_features)
        new_batch['masked_target_mask'].append(masked_target_mask)

        new_batch['masked_source_pts'].append(masked_source_pts)
        new_batch['masked_source_labels'].append(masked_source_labels)
        new_batch['masked_source_features'].append(masked_source_features)
        new_batch['masked_source_mask'].append(masked_source_mask)

    new_batch['masked_target_labels_list'] = new_batch['masked_target_labels']
    new_batch['masked_source_labels_list'] = new_batch['masked_source_labels']
    new_batch['masked_target_mask_list'] = new_batch['masked_target_mask']
    new_batch['masked_source_mask_list'] = new_batch['masked_source_mask']

    for k, i in new_batch.items():
        if k in ['masked_target_pts', 'masked_target_features', \
                 'masked_source_pts', 'masked_source_features', \
                 'masked_source_mask', 'masked_target_mask']:
            new_batch[k] = torch.cat(i, dim=0) # .cuda()
            if not new_batch[k].is_cuda:
                new_batch[k] = new_batch[k].cuda()
        elif '_list' not in k:
            new_batch[k] = torch.cat(i, dim=0)
        else:
            pass
    return new_batch

