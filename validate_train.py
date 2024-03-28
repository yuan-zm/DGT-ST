# Common

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import MinkowskiEngine as ME

import wandb
from tqdm import tqdm
import numpy as np
  
from utils.np_ioueval import iouEval
from utils.avgMeter import AverageMeter

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [N, C]
        dst: target points, [M, C]
    Output:
        dist: per-point square distance, [N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)

    return dist

class validater:
    def __init__(self, 
                 cfg,
                 dataset_name,
                 domain,
                 criterion, 
                 tf_writer, logger):
        self.cfg = cfg
        self.domain = domain 
        self.criterion = criterion
        self.tf_writer = tf_writer
        self.logger = logger
        self.dataset_name = dataset_name

        # init dataset for domain
        if dataset_name == 'SemanticKITTI':
            from dataset.semkitti_test_Sparse_Batch import SemanticKITTI_infer_B  
            self.test_dataset = SemanticKITTI_infer_B(self.cfg, 'test')  # [gen_pselab | test]
        elif dataset_name == 'SynLiDAR':
            from dataset.synlidar_test_Sparse_Batch import SynLiDAR_infer_B    
            self.test_dataset = SynLiDAR_infer_B(self.cfg, 'test')  # [gen_pselab | test]
        elif dataset_name == "SemanticPOSS":
            from dataset.SemanticPoss_test_Sparse_Batch import semPoss_infer_B    
            self.test_dataset = semPoss_infer_B(self.cfg, 'test')  # [gen_pselab | test]
        else:
            raise NotImplementedError('The domain: ** {} ** is not implement now.'.format(dataset_name))

        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.cfg.DATALOADER.VAL_BATCH_SIZE,
                                          num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                          worker_init_fn=my_worker_init_fn,
                                          collate_fn=self.test_dataset.collate_fn,
                                          pin_memory=True,
                                          shuffle=False,
                                          drop_last=False
                                          )
        
    def rolling_predict(self, net_G, old_G, c_iter, src_centers=None, domain='src'):
        torch.cuda.empty_cache()

        self.logger.info('Current iter:{}'.format(c_iter))
        
        if src_centers is not None:
            centers_vectors = src_centers.Proto

        net_G.eval()  # set model to eval mode (for bn and dp)
    
        if old_G is not None:
            old_G.eval()
            tea_iou_calc = iouEval(self.cfg.MODEL_G.NUM_CLASSES)
            tea_iou_calc.reset()

        t_dict = {}  # record logs for wandb & tensorboard

        iou_calc = iouEval(self.cfg.MODEL_G.NUM_CLASSES)
        iou_calc.reset()

        losses = AverageMeter()
    
        # iter_loader = iter(self.test_dataloader)
        tqdm_loader = tqdm(self.test_dataloader,
                           total=len(self.test_dataloader), ncols=50)
        with torch.no_grad():
            # for batch_data in self.tgt_train_loader:
            for batch_idx, batch_data in enumerate(tqdm_loader):

                if batch_idx % 500 == 0:
                    torch.cuda.empty_cache()

                batch_data = self.send_data2GPU(batch_data)
                cloud_inds = batch_data['cloud_inds']

                val_G_in = ME.SparseTensor(batch_data['feats_mink'], batch_data['coords_mink'])

                val_logits_1 = net_G(val_G_in, is_train=False)
                val_logits_1 = val_logits_1.F
                val_logits_1 = val_logits_1[batch_data['inverse_map']]

                if old_G is not None:
                    val_logits_tea = old_G(val_G_in, is_train=False)
                    val_logits_tea = val_logits_tea.F
                    val_logits_tea = val_logits_tea[batch_data['inverse_map']]

                val_labels = batch_data['pc_labs']

                # Processing each scan. 对每一帧单独处理
                left_ind = 0
                for scan_i in range(len(cloud_inds)):
                    # get a single scan
                    pc_i_len = batch_data['s_lens'][scan_i]
                    vo_i_2_lab = val_labels[left_ind: left_ind+pc_i_len]

                    # net G
                    vo_i_2_pc = val_logits_1[left_ind: left_ind+pc_i_len, :]
                    loss = self.criterion(vo_i_2_pc, vo_i_2_lab)
                    losses.update(loss.item(), self.cfg.DATALOADER.VAL_BATCH_SIZE)
                    # cal IoU
                    iou_calc.addBatch(vo_i_2_pc.argmax(dim=1), vo_i_2_lab.long())

                    if old_G is not None:
                        vo_i_tea_pc = val_logits_tea[left_ind: left_ind+pc_i_len, :]
                        # cal IoU
                        tea_iou_calc.addBatch(vo_i_tea_pc.argmax(dim=1), vo_i_2_lab.long())

                    # udpate left index
                    left_ind += pc_i_len 

        t_dict['valid_{0}/all_loss_avg'.format(self.domain)] = losses.avg
      
        sp_mean_iou1, sp_iou_list1 = iou_calc.getIoU()# sp1
        self.logger.info('domain: {}, sp1 IoU:{:.1f}'.format(self.domain, sp_mean_iou1 * 100))
        t_dict['valid_{0}/sp1_IoU'.format(self.domain)] = 100 * sp_mean_iou1
        s_sp_1 = ' \n dec1 IoU: \n'
        for ci, iou_tmp in enumerate(sp_iou_list1):
            cn = self.test_dataset.label_name[ci]
            if ci != 0:
                s_sp_1 += '{}:{:5.2f}|'.format(cn, 100 * iou_tmp)
                t_dict['{0}_sp_net_EC/{1}'.format(self.domain, cn)] = 100 * iou_tmp
        self.logger.info(s_sp_1)

        if old_G is not None:
            sp_mean_iou_tea, sp_iou_list_tea = tea_iou_calc.getIoU()# sp1
            self.logger.info('domain: {}, sp tea IoU:{:.1f}'.format(self.domain, sp_mean_iou_tea * 100))
            t_dict['valid_{0}/sp_tea_IoU'.format(self.domain)] = 100 * sp_mean_iou_tea
            s_sp_tea = ' \n dec tea IoU: \n'
            for ci, iou_tmp in enumerate(sp_iou_list_tea):
                cn = self.test_dataset.label_name[ci]
                if ci != 0:
                    s_sp_tea += '{}:{:5.2f}|'.format(cn, 100 * iou_tmp)
                    t_dict['{0}_sp_net_tea_EC/{1}'.format(self.domain, cn)] = 100 * iou_tmp
            self.logger.info(s_sp_tea)
             
        for k, v in t_dict.items():
            self.tf_writer.add_scalar(k, v, c_iter)
            wandb.log({k: v}, step= c_iter)

        torch.cuda.empty_cache()

        return sp_mean_iou1     
              
    @staticmethod
    def send_data2GPU(batch_data):
        for key in batch_data:  # Target data to gpu
            batch_data[key] = batch_data[key].cuda(non_blocking=True)
        return batch_data