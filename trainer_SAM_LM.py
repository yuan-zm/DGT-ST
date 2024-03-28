# Common
import os
import wandb
import numpy as np
import time
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

# my module
import os.path as osp

from dataset.get_dataloader import get_TV_dl
from network.lr_adjust import adjust_learning_rate
from utils import common as com

from network.domain_mix import laserMix
from validate_train import validater

source_label, target_label = 0, 1

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class SAM_LM_Trainer:
    def __init__(self,
                 cfg,
                 net_G, ema_G,
                 G_optim, 
                 logger, tf_writer, device):

        self.start_iter = 0
        self.ml_info = {'bt_tgt_spIoU': 0}
        self.cfg = cfg
        self.logger = logger
        self.tf_writer = tf_writer
        self.device = device

        self.net_G = net_G
      
        self.ema_G = ema_G
     
        self.G_optim = G_optim
      
        """ Define Loss Function """
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # seg loss
       
        """  get_dataset & dataloader """
        self.init_dataloader()
        self.t_val_iter = self.cfg.TRAIN.T_VAL_ITER
        self.s_val_iter = self.cfg.TRAIN.S_VAL_ITER

        """ Other training parameters"""
        self.c_iter = 0  # Current Iter
        self.round = 0 # current round
        self.best_IoU_iter = 0
        self.best_IoU_after_saveIter = 0
        
        if self.cfg.MEAN_TEACHER.use_mt:
            self.create_ema_model(self.ema_G, self.net_G)
       
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

            # send data to GPU
            src_BData = self.src_TraDL.next()
            self.src_BData = self.send_data2GPU(src_BData)
            self.tgt_BData = self.send_data2GPU(tgt_BData)

            # 1. use teacher model to generate pseudo label
            with torch.no_grad():  # old-model generate pseudo-label
                tgt_G_in = ME.SparseTensor(self.tgt_BData['feats_mink'], self.tgt_BData['coords_mink'])
                self.tgt_o_logits = self.ema_G(tgt_G_in, is_train=False)

            # 2. filter pseudo label with confidence 
            target_confidence_th = self.cfg.PSEUDO_LABEL.threshold # self.target_confidence_th
            target_pseudo = self.tgt_o_logits.F
            target_pseudo = F.softmax(target_pseudo, dim=-1)
            target_conf, target_pseudo = target_pseudo.max(dim=-1)
            filtered_target_pseudo = torch.zeros_like(target_pseudo)
            valid_idx = target_conf > target_confidence_th
            filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
            target_pseudo = filtered_target_pseudo.long()
            self.tgt_BData['pseudo_label'] = target_pseudo
            # mask data
            self.masked_batch = laserMix(self.cfg, self.src_BData, self.tgt_BData)

            # update G
            src_loss = self.train_source()
            tgt_loss = self.train_target()
            all_loss = src_loss + tgt_loss
            all_loss.backward()
            
            self.G_optim.step()

            if self.cfg.MEAN_TEACHER.use_mt and \
                self.cfg.MEAN_TEACHER.alpha_ema > 0 and \
                    self.c_iter % self.cfg.MEAN_TEACHER.update_every == 0:
                self.update_ema_variables(self.ema_G, self.net_G)

            if self.c_iter % self.cfg.TRAIN.LOG_PERIOD == 0:
                print('iter:{0:6d}, '
                      'seg_Ls:{1:.4f}, '
                      'pse_seg_loss:{2:.4f}, '
                      'itr:{3:.3f}, '
                      'Exp:{4}'.format(self.c_iter,
                                       self.wb_dict['netG/seg_Loss'],
                                       self.wb_dict['netG/pse_seg_loss'],
                                       time.time() - start_t,
                                       self.cfg.TRAIN.EXP_NAME))

                self.save_log()  # save logs

            if self.c_iter % self.t_val_iter == 0: # Traget domain val.
                self.valid_and_save()

            if self.c_iter % self.s_val_iter == 0: # Source domain val.
                _  = self.src_valer.rolling_predict(self.net_G, self.ema_G, self.c_iter, domain='src')

            if self.c_iter % 10 == 0:
                torch.cuda.empty_cache()

            if self.c_iter == self.cfg.TRAIN.MAX_ITERS:
                if self.c_iter % self.t_val_iter != 0:
                    self.valid_and_save()
                print("Finish training, this is max iter: {}".format(self.c_iter))
                quit()

        torch.cuda.empty_cache()

    def train_source(self):# ===========train G ================
        if self.cfg.DATASET_SOURCE.use_aug_for_laserMix:
            src_labels = self.src_BData["aug_labels_mink"].cuda()
            src_G_in = ME.SparseTensor(coordinates=self.src_BData["aug_coords_mink"].int(),
                                          features=self.src_BData["aug_feats_mink"])
        else:
            src_labels = self.src_BData["labels_mink"].cuda()
            src_G_in = ME.SparseTensor(coordinates=self.src_BData["coords_mink"].int(),
                                          features=self.src_BData["feats_mink"])
                
        # Train with Source. compute source seg loss        
        self.src_logits = self.net_G(src_G_in)

        all_src_loss = 0.
      
        # loss 1. main classifier CE loss
        src_seg_loss = self.criterion(self.src_logits.F, src_labels)
        all_src_loss = all_src_loss + src_seg_loss
        
        self.wb_dict['netG/seg_Loss'] = src_seg_loss.mean()
        self.wb_dict['netG/all_src_loss'] = all_src_loss.mean()
        
        return all_src_loss
    
    def train_target(self):
        all_tgt_loss = 0.
        t2s_stensor = ME.SparseTensor(coordinates=self.masked_batch["masked_source_pts"].int(),
                                      features=self.masked_batch["masked_source_features"])
        self.t2s_out = self.net_G(t2s_stensor)
        t2s_labels = self.masked_batch["masked_source_labels"].cuda()
        t2s_loss = self.criterion(self.t2s_out.F, t2s_labels.long())
        all_tgt_loss = all_tgt_loss + t2s_loss
        self.wb_dict['netG/pse_seg_loss'] = t2s_loss.mean()

        # kl loss
        if self.cfg.TGT_LOSS.lambda_sac > 0:
            with torch.no_grad():  # old-model generate pseudo-label
                tea_tgt_G_in = ME.SparseTensor(self.tgt_BData['feats_mink'], self.tgt_BData['coords_mink'])
                tea_raw_tgt_logit = self.ema_G(tea_tgt_G_in)
                del_tgt_out = tea_raw_tgt_logit.F[self.tgt_BData['aug_del_mask']]
                raw2aug_tgt_out = del_tgt_out[self.tgt_BData['aug_unique_map']]
            
            # change raw tgt to aug
            tgt_G_in = ME.SparseTensor(self.tgt_BData['aug_feats_mink'], self.tgt_BData['aug_coords_mink'])
            stu_aug_tgt_logit = self.net_G(tgt_G_in)
            
            sac_loss = F.kl_div(F.log_softmax(stu_aug_tgt_logit.F, dim=1),
                                F.softmax(raw2aug_tgt_out.detach(), dim=1))
            sac_loss = sac_loss * self.cfg.TGT_LOSS.lambda_sac 
            all_tgt_loss = all_tgt_loss + sac_loss
            self.wb_dict['netG/sac_loss'] = sac_loss.mean()

        return all_tgt_loss

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

        tgt_sp_iou = self.tgt_valer.rolling_predict(self.net_G, self.ema_G, self.c_iter, domain='tgt')

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

            com.save_best_check(self.net_G, None, 
                                self.G_optim, None, None,
                                self.c_iter, self.logger,
                                self.cfg.TRAIN.MODEL_DIR, name=s_name,
                                iou=tgt_sp_iou)

        torch.cuda.empty_cache()

    def save_log(self):
        self.wb_dict['lr/lr_G'] = self.G_optim.state_dict()['param_groups'][0]['lr']

        for k, v in self.wb_dict.items():
            self.tf_writer.add_scalar(k, v, self.c_iter)
            wandb.log({k: v}, step=self.c_iter)

    def set_zero_grad(self):
        self.net_G.train()  # set model to training mode
       
        self.G_optim.zero_grad()

        self.ema_G.eval()
     
    def set_lr(self):
        current_lr_G = adjust_learning_rate(self.cfg.OPTIMIZER.LEARNING_RATE_G,
                                            self.c_iter, self.cfg.TRAIN.MAX_ITERS,
                                            self.cfg.TRAIN.PREHEAT_STEPS)
   
        for index in range(len(self.G_optim.param_groups)):
            self.G_optim.param_groups[index]['lr'] = current_lr_G
     
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
                            self.net_G, None, 
                            self.G_optim, None, 
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
        
        