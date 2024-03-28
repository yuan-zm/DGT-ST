
from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import yaml
import os
"""
这份主要是检查 新得到的beam-label 然后经过downsample 是否和之前投稿的结果一致

"""
beam_label_path = 'change_data/SynLiDAR_beam_label'
seq_list = ['00', '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12']
target_data_name = 'SemanticPOSS'

data_list = []
for seq_id in seq_list:
    seq_path = join(beam_label_path, seq_id)
    new_data = [(seq_id, f) for f in np.sort(os.listdir(seq_path))]
    data_list.extend(new_data)
    
print(data_list)

def get_specific_beam_mask(beamLabel, choseLabel):
    mask = np.zeros((beamLabel.shape[0])).astype(np.bool)
    for i in range(0, len(choseLabel)):
        save_label_mask = beamLabel == choseLabel[i]
        mask[save_label_mask] = True

    return mask

for pc_name in data_list:
    seq_id, frame_id = pc_name[0], pc_name[1]
    ori_downSampleIds = os.path.expanduser('~/dataset/SynLidar_DownSampled_Data/dataDownSample_Index/To_POSS')
    beamLabel_inds_path = join(ori_downSampleIds, seq_id, frame_id)
    ori_saveBeamLabel_inds = np.load(beamLabel_inds_path)
            
    # down sample to poss
    temp_beam_label_path = join(beam_label_path, seq_id, frame_id)
    beam_label = np.load(temp_beam_label_path)
    
    chose_beam_label = np.arange(0, 64)
    choose_idxs = chose_beam_label[::2] # chose 32 beams first
    rest_choose_idxs = chose_beam_label[1::2][:8] # chose another 8 beams near the LiDAR center
    choose_idxs = np.concatenate((choose_idxs, rest_choose_idxs), axis=0)

    beam_label_to_poss = get_specific_beam_mask(beam_label, choose_idxs)
    
    if np.unique(ori_saveBeamLabel_inds == np.where(beam_label_to_poss==True)[0]):
        print('ok, this is {}'.format(pc_name))
    else:
        print('Error, this is {}'.format(pc_name))
        quit()