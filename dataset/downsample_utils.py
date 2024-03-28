import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
import cv2
from torch.nn import functional as F
import torch
import pdb

def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)
    theta = (theta / np.pi) * 180

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)
    phi_ = (phi_ / np.pi) * 180

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)
    phi = (phi / np.pi) * 180

    phi[phi_ < 0] = 360 - phi[phi_ < 0]
    phi[phi == 360] = 0

    return theta, phi

def beam_label(theta, beam):
    estimator = KMeans(n_clusters=beam)
    res = estimator.fit_predict(theta.reshape(-1, 1))
    label = estimator.labels_
    centroids = estimator.cluster_centers_
    return label, centroids[:,0]

# def change_beam_label(old_beam_label, idxs):
#     new_beam_label = np.zeros((old_beam_label.shape[0])).astype(np.int32)

#     for i in range(len(idxs)):
#         mask_i = (old_beam_label == idxs[i])
#         new_beam_label[mask_i] = i

#     return new_beam_label


# def generate_choosed_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
#     mask = np.zeros((phi.shape[0])).astype(np.bool)

#     for i in range(len(idxs)):
#         phi_i = phi[label == idxs[i]]
#         idxs_phi = np.argsort(phi_i)
#         mask_i = (label == idxs[i])
#         mask_temp = np.zeros((phi_i.shape[0])).astype(np.bool)
#         mask_temp[idxs_phi[::bin_ratio]] = True
#         mask[mask_i] = mask_temp

#     return mask

def get_specific_beam_mask(beamLabel, choseLabel):
    mask = np.zeros((beamLabel.shape[0])).astype(np.bool)
    for i in range(0, len(choseLabel)):
        save_label_mask = beamLabel == choseLabel[i]
        mask[save_label_mask] = True

    return np.where(mask == True)[0]

def generate_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
    mask = np.zeros((phi.shape[0])).astype(np.bool)

    for i in range(0, beam, beam_ratio):
        phi_i = phi[label == idxs[i]]
        idxs_phi = np.argsort(phi_i)
        mask_i = (label == idxs[i])
        mask_temp = np.zeros((phi_i.shape[0])).astype(np.bool)
        mask_temp[idxs_phi[::bin_ratio]] = True
        mask[mask_i] = mask_temp

    return mask