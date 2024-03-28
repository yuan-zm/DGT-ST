# Common
import torch.nn.functional as F
import MinkowskiEngine as ME

from utils.avgMeter import AverageMeter
from network.minkunet import MinkUNet34
from utils.np_ioueval import iouEval
from network.lovasz_losses import Lovasz_loss

# config file
from configs.config_base import cfg_from_yaml_file
from easydict import EasyDict

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import datetime
from tqdm import tqdm
import numpy as np
import argparse
import warnings
import logging
import os
import shutil

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None,
                    help='Model checkpoint path [default: None]')
parser.add_argument('--config-file', default='configs/sourceOnly_nusc2sk_10.yaml',  # multi_scale_D20cm
                    help='config file [default: syn2sk]')
parser.add_argument('--max_epoch', type=int, default=60,
                    help='Epoch to run [default: 100]')

# arguments for ddp training
parser.add_argument('--use_ddp', type=int, default=1,
                    help='using ddp to train model')
parser.add_argument('--local_rank', type=int, default=0,
                    help="If using ddp to train")


FLAGS = parser.parse_args()

cfg = EasyDict()
cfg_from_yaml_file(FLAGS.config_file, cfg)

if cfg.TRAIN.DEBUG:
    cfg.TRAIN.EXP_NAME = 'debug'

FLAGS.log_dir = './logs/' + cfg.TRAIN.PROJECT_NAME + '/' + cfg.TRAIN.EXP_NAME + '/' + \
    datetime.datetime.now().strftime("%Y-%-m-%d-%H_%M")

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# 1) F T 1h20m  2) T T 1h 3) T F 1h 4) FF
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class Trainer:
    def __init__(self):
        # Init Logging
        if FLAGS.local_rank == 0:
            if not os.path.exists(FLAGS.log_dir):
                os.makedirs(FLAGS.log_dir)
                os.makedirs(FLAGS.log_dir+'/files')
            self.log_dir = FLAGS.log_dir
            log_fname = os.path.join(FLAGS.log_dir, 'log_train.txt')
            LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
            DATE_FORMAT = '%Y%m%d %H:%M:%S'
            logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT,
                                datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Trainer")
        # tensorboard writer
        if FLAGS.local_rank == 0 and not os.path.exists(os.path.join(FLAGS.log_dir, 'tb')):
            os.mkdir(os.path.join(FLAGS.log_dir, 'tb'))
            self.tf_writer = SummaryWriter(os.path.join(self.log_dir, 'tb'))
        if FLAGS.local_rank == 0:
            shutil.copy('train.py',
                        os.path.join(FLAGS.log_dir, 'files', 'train.py'))
            shutil.copy(FLAGS.config_file,
                        os.path.join(FLAGS.log_dir, 'files', FLAGS.config_file.split('/')[-1]))
            shutil.copy('network/minkunet.py',
                        os.path.join(FLAGS.log_dir, 'files', 'minkunet.py'))
            shutil.copy('network/resnet.py',
                        os.path.join(FLAGS.log_dir, 'files', 'resnet.py'))
            if cfg.DATASET_SOURCE.TYPE == 'SynLiDAR':
                shutil.copy('dataset/SynLiDAR_train_Sparse.py',
                            os.path.join(FLAGS.log_dir, 'files', 'SynLiDAR_train_Sparse.py'))
            
            if cfg.DATASET_SOURCE.TYPE == 'SemanticKITTI':
                shutil.copy('dataset/semanticKITTI_train_Sparse_10.py',
                        os.path.join(FLAGS.log_dir, 'files', 'semanticKITTI_train_Sparse_10.py'))

        if FLAGS.use_ddp > 0:
            print('local rank', FLAGS.local_rank)
            torch.cuda.set_device(FLAGS.local_rank)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            device = torch.device(f'cuda:{FLAGS.local_rank}')
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Don't use ddp to train....")

        # get_dataset & dataloader
        if cfg.DATASET_SOURCE.TYPE=='SynLiDAR':
            from dataset.SynLiDAR_train_Sparse import SynLiDAR_dataset

            train_dataset = SynLiDAR_dataset(cfg, 'training')
            val_dataset = SynLiDAR_dataset(cfg, 'validation')
        
        if cfg.DATASET_SOURCE.TYPE == 'SemanticKITTI':
            from dataset.semanticKITTI_train_Sparse_10 import SemanticKITTI_dataset

            train_dataset = SemanticKITTI_dataset(cfg, 'training')
            val_dataset = SemanticKITTI_dataset(cfg, 'validation')

        self.get_tra_val_dataloader(train_dataset, val_dataset)

        # train_dataset.get_class_weight()

        # Network & Optimizer
        self.net = MinkUNet34(3, cfg.MODEL_G.NUM_CLASSES)
        if FLAGS.use_ddp > 0:
            self.net.to(device=FLAGS.local_rank)
        else:
            self.net = self.net.to(device=device)
        # Load the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=cfg.OPTIMIZER.LEARNING_RATE_G)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.98)

        # Load module
        self.highest_sp_val_iou, self.highest_fu_val_iou = 0, 0
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']

        if not FLAGS.use_ddp:
            self.net.to(device)
        else:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(
                self.net)
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[FLAGS.local_rank],
                                                                 find_unused_parameters=True)
            self.net_single = self.net.module

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.lovasz_loss = Lovasz_loss(ignore=0)

        # Multiple GPU Training
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
            if not FLAGS.use_ddp:
                self.net = nn.DataParallel(self.net)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger.info(cfg.TRAIN.EXP_NAME)

    def train_one_epoch(self):
        self.net.train()  # set model to training mode
        losses = AverageMeter()        
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader), ncols=50, desc=cfg.TRAIN.EXP_NAME)

        for batch_idx, batch_data in enumerate(tqdm_loader):
            batch_data = self.send_data2GPU(batch_data)

            self.optimizer.zero_grad()

            all_loss = 0.
            end_points = self.net(batch_data)
            loss = self.criterion(end_points['sp_logits'], end_points['labels_mink'])
            all_loss = all_loss + loss
            if cfg.SRC_LOSS.lambda_lov > 0:
                lov_src_loss = self.lovasz_loss(F.softmax(end_points['sp_logits'], dim=1), end_points["labels_mink"])
                all_loss = loss + lov_src_loss
           
            all_loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), cfg.DATALOADER.TRA_BATCH_SIZE)
            if (FLAGS.local_rank == 0 and FLAGS.use_ddp) or not FLAGS.use_ddp:
                self.tf_writer.add_scalar('train/ce_loss', loss.item(), self.cur_epoch * len(self.train_loader) + batch_idx)
                if cfg.SRC_LOSS.lambda_lov > 0:
                    self.tf_writer.add_scalar('train/lov_loss', lov_src_loss.item(), self.cur_epoch * len(self.train_loader) + batch_idx)

        torch.cuda.empty_cache()

    def train(self):
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))
            if FLAGS.use_ddp:
                self.train_sampler.set_epoch(epoch)

            self.train_one_epoch()
            self.scheduler.step()

            if (FLAGS.local_rank == 0 and FLAGS.use_ddp) or not FLAGS.use_ddp:
                self.logger.info('**** EVAL EPOCH %03d ****' % epoch)
                checkpoint_file = os.path.join(self.log_dir, 'checkpoint.tar')
                self.save_checkpoint(checkpoint_file)
                sp_mean_iou = self.validate()

                # Save best checkpoint
                if sp_mean_iou > self.highest_sp_val_iou:
                    self.highest_sp_val_iou = sp_mean_iou
                    self.logger.info(
                        '**** Best sp mean val iou:{:.1f} ****'.format(self.highest_sp_val_iou * 100))
                    checkpoint_file = os.path.join(self.log_dir, 'checkpoint_val_Sp.tar')
                    self.save_checkpoint(checkpoint_file)

                if self.cur_epoch % 10 == 0:
                    checkpoint_file = os.path.join(self.log_dir, 'checkpoint_epoch_{}.tar'.format(self.cur_epoch))
                    self.save_checkpoint(checkpoint_file)
                if self.cur_epoch % (9 + 1) == 0:
                    checkpoint_file = os.path.join(self.log_dir, 'checkpoint_epoch_{}.tar'.format(self.cur_epoch))
                    self.save_checkpoint(checkpoint_file)

    def validate(self):
        self.net.eval()  # set model to eval mode (for bn and dp)

        sp_iou_calc = iouEval(cfg.MODEL_G.NUM_CLASSES)
        sp_iou_calc.reset()

        losses = AverageMeter()
        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader), ncols=50)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):

                batch_data = self.send_data2GPU(batch_data)

                if FLAGS.use_ddp:
                    end_points = self.net_single(batch_data)
                else:
                    end_points = self.net(batch_data)

                loss = self.criterion(end_points['sp_logits'], end_points['labels_mink'])
                losses.update(loss.item(), cfg.DATALOADER.TRA_BATCH_SIZE)

                sp_iou_calc.addBatch(
                    end_points['sp_logits'].argmax(dim=1),
                    end_points['labels_mink'].long()
                    )

        self.tf_writer.add_scalar('valid/all_loss_avg', losses.avg, self.cur_epoch)

        # sp
        sp_mean_iou, sp_iou_list = sp_iou_calc.getIoU()
        self.tf_writer.add_scalar('valid/all_sp_IoU', 100 * sp_mean_iou, self.cur_epoch)
        self.logger.info('sp IoU:{:.1f}'.format(sp_mean_iou * 100))

        s = ' \n sp IoU: \n'
        for ci, iou_tmp in enumerate(sp_iou_list):
            s += '{:5.2f} '.format(100 * iou_tmp)
            class_name = self.train_dataset.label_name[ci]
            s += ' ' + class_name + ' '
            self.tf_writer.add_scalar('sp/' + class_name, 100 * iou_tmp, self.cur_epoch)

        self.logger.info(s)
        torch.cuda.empty_cache()
        return sp_mean_iou

    @staticmethod
    def send_data2GPU(batch_data):
        for key in batch_data:  # send data to gpu
            batch_data[key] = batch_data[key].cuda(non_blocking=True)
        return batch_data

    def save_checkpoint(self, fname):
        save_dict = {
            # after training one epoch, the start_epoch should be epoch+1
            'epoch': self.cur_epoch+1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)

    def get_tra_val_dataloader(self, train_dataset, val_dataset):
        if not FLAGS.use_ddp:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.DATALOADER.TRA_BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                worker_init_fn=my_worker_init_fn,
                collate_fn=train_dataset.collate_fn,
                pin_memory=True
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.DATALOADER.VAL_BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                worker_init_fn=my_worker_init_fn,
                collate_fn=val_dataset.collate_fn,
                pin_memory=True
            )
        else:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.DATALOADER.TRA_BATCH_SIZE,

                num_workers=cfg.DATALOADER.NUM_WORKERS,
                worker_init_fn=my_worker_init_fn,
                collate_fn=train_dataset.collate_fn,
                pin_memory=True,
                sampler=self.train_sampler
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.DATALOADER.VAL_BATCH_SIZE,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                worker_init_fn=my_worker_init_fn,
                collate_fn=val_dataset.collate_fn,
                pin_memory=True,
                sampler=valid_sampler
            )

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    main()
