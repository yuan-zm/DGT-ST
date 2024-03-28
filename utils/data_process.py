from os.path import join
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm

class DataProcessing:

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]
        return points, remissions

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)
    
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def get_file_list(dataset_path, seq_list):
        data_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            new_data = [(seq_id, f[:-4]) for f in np.sort(os.listdir(pc_path))]
            data_list.extend(new_data)

        return data_list

    def get_active_list(list_root):
        train_list = []
        pool_list = []
        with open(join(list_root, 'label_data.json')) as f:
            train_list = json.load(f)
        with open(join(list_root, 'ulabel_data.json')) as f:
            pool_list = json.load(f)
        pool_list += train_list
        train_list = []
        return train_list, pool_list

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list
    
    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(data_root, paths, num_of_class, remap_lut):
        # pre-calculate the number of points in each category
        num_per_class = [0 for _ in range(num_of_class)]
        # import yaml
        # DATA = yaml.safe_load(open('utils/semantic-kitti.yaml', 'r'))
        # remap_dict = DATA["learning_map"]
        # max_key = max(remap_dict.keys())
        # remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        # remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        for file_path in tqdm(paths, total=len(paths), ncols=50):
            # load labels
            label_path = join(data_root, file_path[0], 'labels', file_path[1] + '.label')
            # if all goes well, open label
            label = np.fromfile(label_path, dtype=np.int32)
            label = label.reshape((-1))
            # label = label & 0xFFFF  # semantic label in lower half
            label = remap_lut[label]
            inds, counts = np.unique(label, return_counts=True)
            for i, c in zip(inds, counts):
                # if i == 0:      # 0 : unlabeled
                #     continue
                # else:
                num_per_class[i] += c
        """
        SynLidar
        array([ 70078303,  27840689,   1757884,   4554586,  37126869,  27441503,
                12322023,   9518655,  14930773, 803315543,   4626117, 248176038,
                10189253, 290481395,  45798303, 132139914,  14352718, 169821087,
                19505585,   4346016])
                
        """
        num_per_class = np.array(num_per_class)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)
