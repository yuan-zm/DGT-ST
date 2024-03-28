
import os
import wandb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from trainer_utils import prob2Ent

from dataset.get_dataloader import get_TV_dl
from network.lr_adjust import adjust_learning_rate, adjust_learning_rate_D
from utils import common as com

from validate_train import validater

from utils.classFeature import prototype_dist_estimator

source_label, target_label = 0, 1

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class PCAN_Trainer:
    def __init__(self,
                 cfg,
                 net_G, old_G, net_D, 
                 G_optim,  D_optim, 
                 logger, tf_writer, device):

        self.start_iter = 0
        self.ml_info = {'bt_tgt_spIoU': 0}
        self.cfg = cfg
        self.logger = logger
        self.tf_writer = tf_writer
        self.device = device

        self.net_G = net_G
        self.net_D = net_D

        self.old_G = old_G
     
        self.G_optim = G_optim
        self.D_optim = D_optim

        """ Define Loss Function """
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # seg loss
        if self.cfg.MODEL_D.GAN_MODE == 'ls_gan':# gan loss
            self.criterionGAN = nn.MSELoss(reduction='none')  
        elif self.cfg.MODEL_D.GAN_MODE == 'vanilla_gan':
            self.criterionGAN = nn.BCEWithLogitsLoss(reduction='none') 

        """  get_dataset & dataloader """
        self.init_dataloader()
        self.t_val_iter = self.cfg.TRAIN.T_VAL_ITER
        self.s_val_iter = self.cfg.TRAIN.S_VAL_ITER

        """ Other training parameters"""
        self.c_iter = 0  # Current Iter
        self.round = 0 # current round
        self.best_IoU_iter = 0
        self.best_IoU_after_saveIter = 0

        self.out_class_center = prototype_dist_estimator(cfg, 96)
        
        if self.cfg.MEAN_TEACHER.use_mt:
            self.create_ema_model(self.old_G, self.net_G)
       
    def train(self):
        for epoch in range(self.cfg.TRAIN.MAX_EPOCHS):
            print("This is epoch: {}".format(epoch))

            self.train_one_epoch()
    
    def train_one_epoch(self):
        for tgt_BData in self.tgt_train_loader:          
            self.wb_dict = {}
            self.c_iter += 1
            start_t = time.time()
            self.set_lr()
            self.set_zero_grad()

            # """ 初始化预测伪标签的模型 Init old_G for generate pseudo-label""" 
            if self.cfg.MEAN_TEACHER.round_change and \
                 (self.c_iter % self.cfg.MEAN_TEACHER.round_period == 0 or \
                    self.c_iter == self.cfg.TRAIN.PREHEAT_STEPS):  # start new round
                
                self.create_ema_model(self.old_G, self.net_G)

            # send data to GPU
            src_BData = self.src_TraDL.next()
            self.src_BData = self.send_data2GPU(src_BData)
            self.tgt_BData = self.send_data2GPU(tgt_BData)

            # update G
            self.train_source()
            self.train_target()
            self.G_optim.step()

            # update D
            self.train_net_D()
            self.D_optim.step()

            # update prototype
            self.update_proto()

            if self.cfg.MEAN_TEACHER.use_mt and \
                self.cfg.MEAN_TEACHER.alpha_ema > 0 and \
                    self.cfg.MEAN_TEACHER.round_change == False:
                self.update_ema_variables(self.old_G, self.net_G)

            if self.c_iter % self.cfg.TRAIN.LOG_PERIOD == 0:
                print('iter:{0:6d}, '
                      'seg_Ls:{1:.4f}, '
                      'adv_Ls:{2:.4f}, '
                      'itr:{3:.3f}, '
                      'Exp:{4}'.format(self.c_iter,
                                       self.wb_dict['netG/seg_Loss'],
                                       self.wb_dict['netG/adv_Loss'],
                                       time.time() - start_t,
                                       self.cfg.TRAIN.EXP_NAME))

            self.save_log()  # save logs

            if self.c_iter % self.t_val_iter == 0: # Traget domain val.
                self.valid_and_save()

            if self.c_iter % self.s_val_iter == 0: # Source domain val.
                _  = self.src_valer.rolling_predict(self.net_G, self.old_G, self.c_iter, domain='src')

            if self.c_iter % 10 == 0:
                torch.cuda.empty_cache()

            if self.c_iter == self.cfg.TRAIN.MAX_ITERS:
                if self.c_iter % self.t_val_iter != 0:
                    self.valid_and_save()
                print("Finish training, this is max iter: {}".format(self.c_iter))
                quit()

        torch.cuda.empty_cache()

    def train_source(self):# ===========train G ================
        # Train with Source. compute source seg loss
        src_G_in = ME.SparseTensor(self.src_BData['aug_feats_mink'], self.src_BData['aug_coords_mink'])
        
        self.src_logits, self.src_outFt = self.net_G(src_G_in)

        all_src_loss = 0.
      
        # loss 1. main classifier CE loss
        src_seg_loss = self.criterion(self.src_logits.F, self.src_BData['aug_labels_mink'])
        all_src_loss = all_src_loss + src_seg_loss
        
        all_src_loss.backward()
        self.wb_dict['netG/seg_Loss'] = src_seg_loss.mean()
        self.wb_dict['netG/all_src_loss'] = all_src_loss.mean()
       
    def train_target(self):
        """ stu-model forward  """ 
        tgt_G_in = ME.SparseTensor(self.tgt_BData['feats_mink'], self.tgt_BData['coords_mink'])
        # tgt_G_in_4 = ME.SparseTensor(self.tgt_BData['feats_mink_4'], self.tgt_BData['coords_mink_4'])
        self.tgt_n_logits, self.tgt_outFt = self.net_G(tgt_G_in)
        
        # 1. 得到伪标签
        if self.cfg.TGT_LOSS.CATEGORY_ADV:
            if self.cfg.MEAN_TEACHER.use_mt and self.cfg.MEAN_TEACHER.round_change == False:
                pse_label, tgt_ps_weight = self.gen_tgt_pse_label_by_MT()
                tgt_ps_weight = 1.
                self.tgt_ps_lab = pse_label
          
        all_tgt_loss = 0
        # loss 1: adv loss 
        adv_loss = self.train_adv()
        all_tgt_loss = all_tgt_loss + adv_loss
     
        all_tgt_loss.backward()

    def gen_tgt_pse_label_by_FixModel(self):
        with torch.no_grad():  # old-model generate pseudo-label
            tgt_G_in = ME.SparseTensor(self.tgt_BData['feats_mink'], self.tgt_BData['coords_mink'])
            use_domain = 'tgt'
            self.tgt_o_logits, self.tgt_Tremis = self.old_G(tgt_G_in, is_train=False)
            # Get the pseudo label
            mask_ent = None
            tgt_n_softM = F.softmax(self.tgt_n_logits.F, 1)
            tgt_o_softM = F.softmax(self.tgt_o_logits.F, 1)
            tgt_n_ent = prob2Ent(tgt_n_softM).sum(1)
            tgt_o_ent = prob2Ent(tgt_o_softM).sum(1)

            mask_n_ent = tgt_n_ent < 0.05
            mask_o_ent = tgt_o_ent < 0.05
            # mask_ent = mask_n_ent * mask_o_ent

            proto_logit = F.cosine_similarity(self.tgt_outFt.unsqueeze(1).detach(), self.out_class_center.Proto.unsqueeze(0), dim=-1)
            mask_proto = proto_logit.max(dim=1)[1] == tgt_ps_lab

            mask_ent = mask_n_ent * mask_o_ent * mask_proto

            self.wb_dict['netG/mask_entSum'] = mask_ent.sum()

            tgt_ps_lab = tgt_o_softM.argmax(dim=1)
            conf = tgt_o_softM.max(dim=1)[0]
            mask = conf.ge(self.cfg.PSEUDO_LABEL.threshold)
            tgt_ps_lab = tgt_ps_lab * mask
            tgt_ps_lab = tgt_ps_lab * mask_ent
        
        return tgt_ps_lab 

    def gen_tgt_pse_label_by_MT(self):
        # init pseudo label 
        tgt_G_in = ME.SparseTensor(self.tgt_BData['feats_mink'], self.tgt_BData['coords_mink'])
      
        with torch.no_grad():  # old-model generate pseudo-label
            use_domain = 'tgt'
            self.tgt_mt_logits = self.old_G(tgt_G_in, is_train=False) # , gen_pslab=True
       
            pred_softM = F.softmax(self.tgt_mt_logits.F, 1)
            tp_ps_lab_temp = pred_softM.argmax(dim=1)

            if self.cfg.PSEUDO_LABEL.use_entropy:
                pred_ent = prob2Ent(pred_softM).sum(1).detach()
                mask = pred_ent < self.cfg.PSEUDO_LABEL.ent_threshold

            if self.cfg.PSEUDO_LABEL.use_confidence:
                conf = pred_softM.max(dim=1)[0]
                mask = conf.ge(self.cfg.PSEUDO_LABEL.threshold)
        
            tgt_ps_lab = torch.zeros_like(tp_ps_lab_temp)
            tgt_ps_lab[mask] = tp_ps_lab_temp[mask]

            pse_weight = mask.sum() / mask.shape[0]
            
            self.wb_dict['netG/mask_entSum'] = mask.sum()

        return tgt_ps_lab, pse_weight

    def train_adv(self):  # ===========train G ================
        
        adv_in = self.tgt_n_logits
        D_logit_out = self.net_D(adv_in)

        adv_lab = torch.zeros_like(D_logit_out)
        adv_loss = self.criterionGAN(D_logit_out, adv_lab)

        if self.c_iter > self.cfg.TRAIN.PREHEAT_STEPS and \
                self.cfg.TGT_LOSS.CATEGORY_ADV and \
                    self.c_iter > self.cfg.TGT_LOSS.cal_start_iter:
            
            assert self.tgt_ps_lab is not None
            cal_adv_loss = self.cal_category_adv_loss(adv_loss, self.tgt_ps_lab)

            lambda_cal_adv = self.cfg.TGT_LOSS.lambda_cal_adv
            adv_loss = cal_adv_loss * lambda_cal_adv + adv_loss.mean() * (1 - lambda_cal_adv)
            
            self.wb_dict['netG/cal_adv_Loss'] = cal_adv_loss
        else:
            adv_loss = adv_loss.mean()

        adv_loss = adv_loss * self.cfg.TGT_LOSS.LAMBDA_ADV 
        self.wb_dict['netG/adv_Loss'] = adv_loss
        
        return adv_loss
    
    def train_net_D(self):     # ===========train D================

        for param in self.net_D.parameters():  # Bring back Grads in D
            param.requires_grad = True
        self.D_optim.zero_grad()
        
        # Train with Source
        src_D_in = self.src_logits.detach()

        src_D_out = self.net_D(src_D_in)
        src_d_loss = self.criterionGAN(src_D_out, torch.zeros_like(src_D_out))

        src_d_loss = src_d_loss.mean() # * 0.5
        src_d_loss.backward()

        # Train with target
        tgt_D_in = self.tgt_n_logits.detach() 

        tgt_D_out = self.net_D(tgt_D_in) 
        tgt_d_loss = self.criterionGAN(tgt_D_out, torch.ones_like(tgt_D_out))

        tgt_d_loss = tgt_d_loss.mean() # * 0.5
        tgt_d_loss.backward()

        self.wb_dict['netD/src'] = src_d_loss
        self.wb_dict['netD/tgt'] = tgt_d_loss

    def cal_category_adv_loss(self, adv_loss, vo_lab):
      
        lab = vo_lab
      
        old_unique_lab, old_uni_counts = torch.unique(lab, return_counts=True)

        for i in range(len(old_unique_lab)):
            if old_uni_counts[i] < 50:
                lab[lab == old_unique_lab[i]] = 0
        
        unique_lab, uni_counts = torch.unique(lab, return_counts=True)

        valid_lab_count = 0.
        final_adv_loss = 0.
        for i in unique_lab:
            # V1
            valid_lab_count += 1
            temp_adv_loss = adv_loss[lab == i, :].view(-1)
            
            if i == 0 or (lab == i).sum() < 30:
                temp_adv_loss_mean = temp_adv_loss.mean()
            elif self.cfg.TGT_LOSS.PROTO_REWEIGHT and self.cfg.TGT_LOSS.CAL_out:
                temp_i_ft = self.tgt_outFt[lab == i, :] # .detach()
                temp_proto = self.out_class_center.Proto[i]
                cossim = 1.0 - F.cosine_similarity(temp_proto.expand_as(temp_i_ft), temp_i_ft)
                temp_adv_loss_mean = (temp_adv_loss * cossim).sum()  / cossim.sum()
            else:
                temp_adv_loss_mean = temp_adv_loss.mean()

            final_adv_loss = final_adv_loss + temp_adv_loss_mean
       
        return final_adv_loss

    def update_proto(self):
        if self.cfg.PROTOTYPE.update_domain == "src":
            self.out_class_center.update(self.src_outFt, self.src_BData['aug_labels_mink'])
            
        if self.cfg.PROTOTYPE.update_domain == "tgt":
            self.out_class_center.update(self.tgt_outFt, self.tgt_ps_lab)

    def valid_and_save(self):
        cp_fn = os.path.join(self.cfg.TRAIN.MODEL_DIR, 'cp_current.tar')
        self.fast_save_CP(cp_fn)

        if self.cfg.TGT_LOSS.CAL_out:
            proto_path = os.path.join(self.cfg.TRAIN.MODEL_DIR, 'cp_out_iter_{}.tar'.format(self.c_iter))
            self.out_class_center.save(proto_path)

        # If you want save model checkpoint, set cfg.TRAIN.SAVE_MORE_ITER = True
        if self.c_iter > self.cfg.TRAIN.SAVE_ITER and self.cfg.TRAIN.SAVE_MORE_ITER:
            cp_fn = os.path.join(self.cfg.TRAIN.MODEL_DIR, 'cp_{}_iter.tar'.format(self.c_iter))
            self.fast_save_CP(cp_fn)

        tgt_sp_iou = self.tgt_valer.rolling_predict(self.net_G, self.old_G, self.c_iter, domain='tgt')

        if (tgt_sp_iou > self.best_IoU_after_saveIter and self.c_iter > self.cfg.TRAIN.SAVE_ITER) or \
                tgt_sp_iou > self.ml_info['bt_tgt_spIoU']:
            s_name = 'target_Sp'

            if (tgt_sp_iou > self.best_IoU_after_saveIter and self.c_iter > self.cfg.TRAIN.SAVE_ITER):
                # 由于点云GAN不稳定，有时候好的结果在最开始出现，所以添加这个if
                self.best_IoU_after_saveIter = tgt_sp_iou
                s_name = 'target_Sp_After'

            self.best_IoU_iter = self.c_iter
            self.ml_info['bt_tgt_spIoU'] = tgt_sp_iou
            wandb.run.summary["bt_tgt_spIoU"] = tgt_sp_iou

            com.save_best_check(self.net_G, self.net_D, 
                                self.G_optim, self.D_optim, None,
                                self.c_iter, self.logger,
                                self.cfg.TRAIN.MODEL_DIR, name=s_name,
                                iou=tgt_sp_iou)

        torch.cuda.empty_cache()

    def save_log(self):
        self.wb_dict['lr/lr_G'] = self.G_optim.state_dict()['param_groups'][0]['lr']
        self.wb_dict['lr/lr_D'] = self.D_optim.state_dict()['param_groups'][0]['lr']

        for k, v in self.wb_dict.items():
            self.tf_writer.add_scalar(k, v, self.c_iter)
            wandb.log({k: v}, step=self.c_iter)

    def set_zero_grad(self):
        self.net_G.train()  # set model to training mode
        self.net_D.train()
        
        self.G_optim.zero_grad()

        self.old_G.eval()
      
        for param in self.net_D.parameters():
            param.requires_grad = False

    def set_lr(self):
        current_lr_G = adjust_learning_rate(self.cfg.OPTIMIZER.LEARNING_RATE_G,
                                            self.c_iter, self.cfg.TRAIN.MAX_ITERS,
                                            self.cfg.TRAIN.PREHEAT_STEPS)
        current_lr_D = adjust_learning_rate_D(self.cfg.OPTIMIZER.LEARNING_RATE_D,
                                            self.c_iter, self.cfg.TRAIN.MAX_ITERS,
                                            self.cfg.TRAIN.PREHEAT_STEPS)
        for index in range(len(self.G_optim.param_groups)):
            self.G_optim.param_groups[index]['lr'] = current_lr_G
        for index in range(len(self.D_optim.param_groups)):
            self.D_optim.param_groups[index]['lr'] = current_lr_D

    def update_ema_variables(self, ema_net, net):
        alpha_teacher = min(1 - 1 / (self.c_iter + 1), self.cfg.MEAN_TEACHER.alpha_ema)
        self.cur_alpha_teacher = alpha_teacher
        for ema_param, param in zip(ema_net.parameters(), net.parameters()):
            ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
        for t, s in zip(ema_net.buffers(), net.buffers()):
            if not t.dtype == torch.int64:
                t.data.mul_(alpha_teacher).add_(s.data, alpha=1 - alpha_teacher)

    def create_ema_model(self, ema, net):
        print('create_ema_model G to current iter {}'.format(self.c_iter))
        for param_q, param_k in zip(net.parameters(), ema.parameters()):
            param_k.data = param_q.data.clone()
        for buffer_q, buffer_k in zip(net.buffers(), ema.buffers()):
            buffer_k.data = buffer_q.data.clone()
        ema.eval()
        for param in ema.parameters():
            param.requires_grad_(False)
        for param in ema.parameters():
            param.detach_()

    @staticmethod
    def send_data2GPU(batch_data):
        for key in batch_data:  # send data to gpu
            batch_data[key] = batch_data[key].cuda(non_blocking=True)
        return batch_data

    def fast_save_CP(self, checkpoint_file):
        com.save_checkpoint(checkpoint_file,
                            self.net_G, self.net_D, 
                            self.G_optim, self.D_optim, 
                            None,
                            self.c_iter)
    
    def init_dataloader(self):
        # init source dataloader
        if self.cfg.DATASET_SOURCE.TYPE == "SynLiDAR":
            from dataset.SynLiDAR_trainSet import SynLiDAR_Dataset
            src_tra_dset = SynLiDAR_Dataset(self.cfg, 'training')
            src_val_dset = SynLiDAR_Dataset(self.cfg, 'validation')
        
        self.src_TraDL, self.src_ValDL = get_TV_dl(self.cfg, src_tra_dset, src_val_dset)
        
        if self.cfg.DATASET_TARGET.TYPE == "SemanticKITTI":
            from dataset.semkitti_trainSet import SemanticKITTI
            t_tra_dset = SemanticKITTI(self.cfg, 'training')
            t_val_dset = SemanticKITTI(self.cfg, 'validation')
        elif self.cfg.DATASET_TARGET.TYPE == "SemanticPOSS":
            from dataset.SemanticPoss_trainSet import semPoss_Dataset
            t_tra_dset = semPoss_Dataset(self.cfg, 'training')
            t_val_dset = semPoss_Dataset(self.cfg, 'validation')  
      
        self.tgt_train_loader, _ = get_TV_dl(self.cfg, t_tra_dset, t_val_dset, domain='target')

        # init validater
        self.src_valer = validater(self.cfg, self.cfg.DATASET_SOURCE.TYPE, 'source', self.criterion, self.tf_writer, self.logger)
        self.tgt_valer = validater(self.cfg, self.cfg.DATASET_TARGET.TYPE, 'target', self.criterion, self.tf_writer, self.logger)
        
        