# Common
import sys
import os
from os.path import join
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch.utils.data as torch_data
from torch.utils.data import DataLoader

from tqdm import tqdm

import numpy as np
from utils.data_process import DataProcessing as DP

from dataset.data_utils import get_sk_data

import MinkowskiEngine as ME

from dataset.downsample_utils import get_specific_beam_mask

# config file
from configs.config_base import cfg_from_yaml_file
from easydict import EasyDict

"""
计算synLidar 进行线label丢弃后, density的计算
"""


parser = argparse.ArgumentParser()
parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='~/dataset/SynLiDAR/sub_dataset',
        help='Dataset dir. No Default',
    )
parser.add_argument(
        '--sequences',  # '-l',
        nargs="+",
        default= ['00', '01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12'] ,
        help='evaluated sequences',
    )
parser.add_argument(
        '--data-name', 
        type=str,
        required=False,
        default="SynLiDAR",
        help='The name of dataset. Default is %(default)s',
    )
parser.add_argument(
        '--voxel-size', 
        type=float,
        required=False,
        default=0.05, # 5cm
        help='Voxel size of voxilization. Default is 5cm',
    )

parser.add_argument(
        '--tgt-data-name', 
        type=str,
        required=False,
        default="SemanticPOSS",
        help='The name of the target dataset. Default is %(default)s',
    )
FLAGS = parser.parse_args()

class mini_dataset(torch_data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_path = os.path.expanduser(cfg.data_path)
        seq_list = cfg.sequences
        self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        self.data_list = sorted(self.data_list)
        print('This is ** {} ** dataset, filepath is ** {} ** \n \
                voxel size is ** {} **, has ** {} ** scans.'.
                format(cfg.data_name, self.dataset_path, cfg.voxel_size, len(self.data_list)))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        pc_name = self.data_list[item]
        pc, _, _ = get_sk_data(
                                pc_name,
                                self.dataset_path,
                                None,
                                FLAGS.data_name
                                )
        
        # get the beam label
        beamLabel_path = os.path.expanduser('change_data/SynLiDAR_beam_label')
        seq_id, frame_id = pc_name[0], pc_name[1]
        beamLabel_inds_path = join(beamLabel_path, seq_id, frame_id + '.npy')
        beamLabel = np.load(beamLabel_inds_path)
        
        total_beam_labels = np.arange(0, 64)
        
        if self.cfg.tgt_data_name == 'SemanticPOSS': # 40 beams
            choose_beams = total_beam_labels[::2] # chose 32 beams first
            rest_choose_beams = total_beam_labels[1::2][:8] # chose another 8 beams near the LiDAR center
            choose_beams = np.concatenate((choose_beams, rest_choose_beams), axis=0)
        if self.cfg.tgt_data_name == 'nuScenes': # 32 beams
            choose_beams = total_beam_labels[::2] # only use 32 beams

        choseInds = get_specific_beam_mask(beamLabel, choose_beams)
        pc = pc[choseInds]
      
        _, v_xyz = ME.utils.sparse_quantize(
                                            coordinates=pc,
                                            features=pc ,
                                            quantization_size=FLAGS.voxel_size
                                            )
        return v_xyz
    
    def collate_fn(self, batch):
        v_xyz = []
        for i in range(len(batch)):
            v_xyz.append(batch[i])
        v_xyz_batch = np.vstack(v_xyz)
        
        return v_xyz_batch

cal_dataset = mini_dataset(FLAGS)
cal_dataloader = DataLoader(
                        cal_dataset,
                        batch_size=16,
                        num_workers=4,
                        collate_fn=cal_dataset.collate_fn,
                        shuffle=False,
                        drop_last=False,
                        )

tqdm_cal_dataloader = tqdm(cal_dataloader, total=len(cal_dataloader), ncols=50)

all_num_each_bin = np.zeros(10)

for  batch_idx, batch_v_xyz in enumerate(tqdm_cal_dataloader):
   
    xy_dis = np.sqrt(batch_v_xyz[:, 0]**2 + batch_v_xyz[:, 1]**2)
    num_each_bin, _ = np.histogram(xy_dis ,bins=10, range=(0, 100))
    all_num_each_bin += num_each_bin
    
print((all_num_each_bin / cal_dataset.__len__()).tolist())
print('done')

# SynLidar voxel size = 5cm
# [34404.90337702 21186.50640121  7464.47998992  3618.06023185
#  1907.87081653  1166.47878024   758.06068548   520.30176411
#  365.83371976   266.61421371]