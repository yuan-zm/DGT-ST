# Common
from dataset.semPoss_test_Sparse import semPoss_infer_B    
from network.minkunet import MinkUNet34
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import argparse
import warnings
import logging
import yaml
import os
import MinkowskiEngine as ME

# config file
from configs.config_base import cfg_from_yaml_file
from easydict import EasyDict

'''
预测结果 softmax 然后累加起来
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# trained_model/2021-7-09-16_42/checkpoint_val_Spfuion.tar
np.random.seed(0)
warnings.filterwarnings("ignore")
# ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path',
                    default='logs/syn2sp/Ori_M34_XYZ/2024-2-20-12_49/checkpoint_epoch_10.tar',
                    help='Model checkpoint path [default: None]')
parser.add_argument('--infer_mode', default='test', type=str,
                    help='Predicted sequence id [gen_pselab | test]')
parser.add_argument('--result_dir', default='res_pred/syn2sp/Ori_M34_XYZ_Epoch10/',
                    help='Dump dir to save prediction [default: result/]')
parser.add_argument('--only_pred', default=1, type=int,
                    help='0---> predictions and probs | 1--> only predictions')
parser.add_argument('--yaml_config', default='utils/semantic-kitti.yaml',
                    help='semantic-kitti.yaml path')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch Size during training [default: 30]')
parser.add_argument('--index_to_label', action='store_true', default=False,
                    help='Set index-to-label flag when inference / Do not set it on seq 08')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers [default: 5]')
parser.add_argument('--num_classes', type=int, default=14,
                    help='Number of workers [default: 5]')
parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/sourceOnly_syn2sp.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )

FLAGS = parser.parse_args()

cfg = EasyDict()
cfg.OUTPUT_DIR = './workspace/'
cfg_from_yaml_file(FLAGS.config_file, cfg)
    
cfg.TRAIN.DEBUG = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


FLAGS.debug = False

class Tester:
    def __init__(self):
        # Init Logging
        os.makedirs(FLAGS.result_dir, exist_ok=True)
        log_fname = os.path.join(FLAGS.result_dir, 'log_test.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT,
                            datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Tester")

        self.test_dataset = semPoss_infer_B(cfg, FLAGS.infer_mode)  # [gen_pselab | test]
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=FLAGS.batch_size,
                                          num_workers=FLAGS.num_workers,
                                          worker_init_fn=my_worker_init_fn,
                                          collate_fn=self.test_dataset.collate_fn,
                                          pin_memory=True,
                                          shuffle=False,
                                          drop_last=False
                                          )

        # Network & Optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = MinkUNet34(3, FLAGS.num_classes).to(device)

        # Load module
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            if not os.path.exists(FLAGS.checkpoint_path):
                print('No model !!!!')
                quit()
            print('this is: %s ' % FLAGS.checkpoint_path)
            checkpoint = torch.load(CHECKPOINT_PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'])

        # Multiple GPU Testing
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" %
                             (torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)

        self.saveed_pred_dir = []

    def load_yaml(self, path):
        DATA = yaml.safe_load(open(path, 'r'))
        # get number of interest classes, and the label mappings
        remapdict = DATA["learning_map_inv"]
        # make lookup table for mapping
        maxkey = max(remapdict.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        return remap_lut

    def test(self):
        self.logger.info("Start Testing")
        self.rolling_predict()
        # Merge Probability
        print('wow done')

    def rolling_predict(self):
        self.net.eval()  # set model to eval mode (for bn and dp)
        count = 0
        # iter_loader = iter(self.test_dataloader)
        tqdm_loader = tqdm(self.test_dataloader,
                           total=len(self.test_dataloader))
        with torch.no_grad():
            # for batch_data in self.tgt_train_loader:
            for batch_idx, batch_data in enumerate(tqdm_loader):
                for key in batch_data:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

                cloud_inds = batch_data['cloud_inds']
                
                # Forward pass
                end_points = self.net(batch_data)

                voxel2pc = end_points['sp_logits']
                voxel2pc_F = voxel2pc[end_points['inverse_map']]

                # 对每一帧单独处理
                pc_temp_count = 0
                for scan_i in range(len(cloud_inds)):
                    pc_i_len = batch_data['s_lens'][scan_i]
                    vo_i_2_pc = voxel2pc_F[pc_temp_count: pc_temp_count+pc_i_len, :]
                   
                    soft_pred = F.softmax(vo_i_2_pc, dim=1)
                    
                    pc_temp_count += pc_i_len 
                    # save prediction
                    root_dir = os.path.join(FLAGS.result_dir, self.test_dataset.data_list[cloud_inds[scan_i]][0], 'predictions')
                    if self.test_dataset.data_list[cloud_inds[scan_i]][0] not in self.saveed_pred_dir:
                        self.saveed_pred_dir.append(self.test_dataset.data_list[0][0])
                        os.makedirs(root_dir, exist_ok=True)
                        # vo_prob
                        vo_dir = os.path.join(FLAGS.result_dir, self.test_dataset.data_list[cloud_inds[scan_i]][0], 'vo_prob')
                        os.makedirs(vo_dir, exist_ok=True)

                    pred = np.argmax(soft_pred.cpu().numpy(), 1).astype(np.uint32)
               
                    if FLAGS.index_to_label is True: 
                        name = self.test_dataset.data_list[cloud_inds[scan_i]][1] + '.label'
                        output_path = os.path.join(root_dir, name)
                        pred.tofile(output_path)
                    else:  # 0 - 19
                        name = self.test_dataset.data_list[cloud_inds[scan_i]][1] + '.npy'
                        output_path = os.path.join(root_dir, name)
                        np.save(output_path, pred)
                    if FLAGS.only_pred == 0:
                        output_path_pro = os.path.join(vo_dir, name.replace('.label', '_sp.npy') if 'label' in name else name.replace('.npy', '_sp.npy'))
                        np.save(output_path_pro, end_points['sp_logits'].cpu().numpy())


def main():
    tester = Tester()
    tester.test()


if __name__ == '__main__':
    main()

