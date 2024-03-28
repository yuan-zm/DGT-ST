# Common
import sys
import os
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

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
FLAGS = parser.parse_args()

dataset_path = os.path.expanduser(FLAGS.data_path)
seq_list = FLAGS.sequences

data_list = DP.get_file_list(dataset_path, seq_list)
print('This is ** {} ** dataset, filepath is ** {} ** \n \
       voxel size is ** {} **, has ** {} ** scans.'.
       format(FLAGS.data_name, dataset_path, FLAGS.voxel_size, len(data_list)))

tqdm_data_list = tqdm(data_list, total=len(data_list), ncols=50)

all_num_each_bin = np.zeros(10)

for data_idx, pc_name in enumerate(tqdm_data_list):
    pc, remis, lab = get_sk_data(
                                pc_name,
                                dataset_path,
                                None,
                                FLAGS.data_name
                                )
    
    v_inds, v_xyz = ME.utils.sparse_quantize(
                                            coordinates=pc,
                                            features=pc ,
                                            quantization_size=FLAGS.voxel_size
                                            )
    
    xy_dis = np.sqrt(v_xyz[:, 0]**2 + v_xyz[:, 1]**2)
    num_each_bin, _ = np.histogram(xy_dis ,bins=10, range=(0, 100))
    all_num_each_bin += num_each_bin
    
print(all_num_each_bin / len(tqdm_data_list))
print('done')

# SynLidar voxel size = 5cm
# [34404.90337702 21186.50640121  7464.47998992  3618.06023185
#  1907.87081653  1166.47878024   758.06068548   520.30176411
#  365.83371976   266.61421371]