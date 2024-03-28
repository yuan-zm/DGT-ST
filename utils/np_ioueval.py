#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import torch

class iouEval:
    def __init__(self, n_classes, ignore=[0]):
        # classes
        self.n_classes = n_classes

        # What to include and ignore from the means
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
        # print("[IOU EVAL] IGNORE: ", self.ignore)
        # print("[IOU EVAL] INCLUDE: ", self.include)

        # reset the class counters
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes,
                                    self.n_classes),
                                    dtype=np.int64)

    def addBatch(self, x, y):  # x=preds, y=targets

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # sizes should be matching
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # check
        assert(x_row.shape == x_row.shape)

        # create indexes
        idxs = tuple(np.stack((x_row, y_row), axis=0))

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.conf_matrix, idxs, 1)

    def getStats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.copy()
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"

    def get_confusion(self):
        return self.conf_matrix.copy()
