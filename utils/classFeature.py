import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class prototype_dist_estimator():
    def __init__(self, cfg, feature_num, resume_path=None):
        super(prototype_dist_estimator, self).__init__()

        self.cfg = cfg
        self.IGNORE_LABEL = 0
        self.class_num = cfg.MODEL_G.NUM_CLASSES
        self.feature_num = feature_num
        
        # momentum 
        self.use_momentum = True if cfg.PROTOTYPE.SRCPROTOTYPE_UPDATE_MODE == 'moving_average' else False
        self.momentum = cfg.PROTOTYPE.PROTOTYPE_EMA_STEPLR

        # init prototype
        self.init(feature_num=self.feature_num, resume=resume_path)

    def init(self, feature_num, resume):
        if resume:
            if feature_num == self.cfg.MODEL_G.NUM_CLASSES:
                resume = os.path.join(resume, 'prototype_out_dist.pth')
            elif feature_num == self.feature_num:
                resume = os.path.join(resume, 'prototype_feat_dist.pth')
            else:
                raise RuntimeError("Feature_num not available: {}".format(feature_num))
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.Proto = checkpoint['Proto'].cuda(non_blocking=True)
            self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    def update(self, features, labels=None):
        features = features
       
        if labels == None: # target domain without labels
            # pred = pred.F # .argmax(1)
            pred_argmax = pred.argmax(dim=1)

            # pred_softmax = F.softmax(pred, dim=1)

            # thresh = self.cfg.PROTOTYPE.PSELAB_THRESH
            # conf = pred_softmax.max(dim=1)[0]
            # mask = conf.ge(thresh)

            # labels = pred_argmax * mask
            labels = pred_argmax
        # else:
            # mask = pred_argmax == labels
        mask = labels != self.IGNORE_LABEL

        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        if not self.use_momentum:
            N, A = features.size()
            C = self.class_num
            # refer to SDCA for fast implementation
            features = features.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            features_by_sort = features.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = features_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(
                sum_weight + self.Amount.view(C, 1).expand(C, A)
            )
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
            # 限制最多100000 太大了 更新很慢
            self.Amount[self.Amount > 100000] = 100000
            # self.Amount[i] = min(100000, self.Amount[i] + num_clsi)
        else:
            # momentum implementation
            ids_unique = labels.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = (labels == i)
                feature = features[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = (1 - self.momentum) * feature + self.Proto[i, :] * self.momentum 
        
    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   name)




def class_MMD_alignment(self, ids, vectors):
    sigma_list = [0.01, 0.1, 1, 10, 100]

    loss = torch.Tensor([0]).cuda()
    for i in range(len(ids)):
        if ids[i] in [0]:
            continue
        temp_loss = mix_rbf_mmd2(vectors[i].unsqueeze(0), self.Proto[ids[i]].unsqueeze(0), sigma_list)
        loss = loss + temp_loss

    loss = loss / len(ids)  # * 10
    pass
    return loss
