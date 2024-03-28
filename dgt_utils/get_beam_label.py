# Common
import sys
import os
import argparse
from os.path import join

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from dataset.downsample_utils import compute_angles, beam_label, generate_mask, generate_choosed_mask, change_beam_label

import torch.utils.data as torch_data
from torch.utils.data import DataLoader

from tqdm import tqdm

import numpy as np
from utils.data_process import DataProcessing as DP

from dataset.data_utils import get_sk_data

import MinkowskiEngine as ME

# config file
from configs.config_base import cfg_from_yaml_file
from easydict import EasyDict

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
        '--num-beams', 
        type=int,
        required=False,
        default=64, 
        help='The total number of beams of this dataset',
    )

FLAGS = parser.parse_args()

class mini_dataset(torch_data.Dataset):
    def __init__(self, cfg):
        self.num_beams = cfg.num_beams
        print("This dataset has {} beams.".format(cfg.num_beams))
        self.dataset_path = os.path.expanduser(cfg.data_path)
        seq_list = cfg.sequences
        self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        self.data_list = sorted(self.data_list)
        print('This is ** {} ** dataset, filepath is ** {} ** \n \
                voxel size is ** {} **, has ** {} ** scans.'.
                format(cfg.data_name, self.dataset_path, cfg.voxel_size, len(self.data_list)))
        self.count = 0
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        self.count += 1
        pc_name = self.data_list[item]
        print("Processed {}/{}. This is seq:{}: {}.   ".format(self.count, len(self.data_list), pc_name[0], pc_name[1]), end="\r")
        pc, remis, lab = get_sk_data(
                                pc_name,
                                self.dataset_path,
                                None,
                                FLAGS.data_name
                                )
        pc_np = pc[:, :3]

        theta, phi = compute_angles(pc_np)
        label, centroids = beam_label(theta, self.num_beams)
        idxs = np.argsort(centroids)

        seq_id, frame_id = pc_name[0], pc_name[1]

        new_beam_label = change_beam_label(label, idxs)
        # 保存 线 label
        save_base_path = "change_data/{}_beam_label".format(FLAGS.data_name)
        mask_base_path = join(save_base_path, seq_id)
        os.makedirs(mask_base_path, exist_ok=True)
        beam_label_save_path = join(mask_base_path, frame_id + '.npy')
        
        np.save(beam_label_save_path, new_beam_label)
        return pc
    
    def collate_fn(self, batch):
        v_xyz = []
        for i in range(len(batch)):
            v_xyz.append(batch[i])
        v_xyz_batch = np.vstack(v_xyz)
        
        return v_xyz_batch

cal_dataset = mini_dataset(FLAGS)
cal_dataloader = DataLoader(
                        cal_dataset,
                        batch_size=4,
                        num_workers=8,
                        collate_fn=cal_dataset.collate_fn,
                        shuffle=False,
                        drop_last=False,
                        )

tqdm_cal_dataloader = tqdm(cal_dataloader, total=len(cal_dataloader), ncols=50)

all_num_each_bin = np.zeros(10)

for  batch_idx, batch_v_xyz in enumerate(tqdm_cal_dataloader):
   
    # xy_dis = np.sqrt(batch_v_xyz[:, 0]**2 + batch_v_xyz[:, 1]**2)
    # num_each_bin, _ = np.histogram(xy_dis ,bins=10, range=(0, 100))
    # all_num_each_bin += num_each_bin
    pass
    
print((all_num_each_bin / cal_dataset.__len__()).tolist())
print('done')

# SynLidar voxel size = 5cm
# [34404.90337702 21186.50640121  7464.47998992  3618.06023185
#  1907.87081653  1166.47878024   758.06068548   520.30176411
#  365.83371976   266.61421371]