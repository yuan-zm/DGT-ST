# Common
import os

'''
预测结果 softmax 然后累加起来
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
# my module

from network.RandLANet import Network
from dataset.semkitti_test_Sparse_Grid import SemanticKITTI


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# trained_model/2021-7-09-16_42/checkpoint_val_Spfuion.tar
np.random.seed(0)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()  # ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
parser.add_argument('--checkpoint_path',
                    default='logs/SingleScale_Znorm/2021-11-30-01_11/checkpoint_val_Sp.tar',
                    help='Model checkpoint path [default: None]')
parser.add_argument('--infer_mode', default='test', type=str,
                    help='Predicted sequence id [gen_pselab | test]')
parser.add_argument('--result_dir', default='singleScale_ZNorm_Grid/',
                    help='Dump dir to save prediction [default: result/]')
parser.add_argument('--only_pred', default=0, type=int,
                    help='0---> predictions and probs | 1--> only predictions')
parser.add_argument('--yaml_config', default='utils/semantic-kitti.yaml', help='semantic-kitti.yaml path')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 30]')
parser.add_argument('--index_to_label', action='store_true', default=False,
                    help='Set index-to-label flag when inference / Do not set it on seq 08')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of workers [default: 5]')
FLAGS = parser.parse_args()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


class Tester:
    def __init__(self):
        # Init Logging
        os.makedirs(FLAGS.result_dir, exist_ok=True)
        log_fname = os.path.join(FLAGS.result_dir, 'log_test.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Tester")

        # load yaml file
        # self.remap_lut = self.load_yaml(FLAGS.yaml_config)

        # get_dataset & dataloader
        # self.test_dataset = SemanticKITTI('test', test_id=FLAGS.test_id, batch_size=FLAGS.batch_size)
        self.test_dataset = SemanticKITTI(FLAGS.infer_mode, once_infer=True, infer_bs=FLAGS.batch_size)  #[gen_pselab | test]
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
        self.net = Network(cfg, cfg.num_classes).to(device)

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
            self.logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
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
        tqdm_loader = tqdm(self.test_dataloader, total=len(self.test_dataloader))
        with torch.no_grad():
            # for batch_data in self.tgt_train_loader:
            for batch_idx, batch_data in enumerate(tqdm_loader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                cloud_inds = batch_data['cloud_inds']
                input_inds = batch_data['selected_idx']
                # Forward pass
                # torch.cuda.synchronize()
                end_points = self.net(batch_data)#, domain='target') # 输出为 B C N
                # 因为输入之前shuffle了一次 所以要翻转顺序
                end_points['sp_fusion_logits'].scatter_(2, input_inds.unsqueeze(1).repeat(1, cfg.num_classes, 1), end_points['sp_fusion_logits'].clone())
                end_points['sp_logits'].scatter_(2, input_inds.unsqueeze(1).repeat(1, cfg.num_classes, 1), end_points['sp_logits'].clone())

                # 按照batch维度累加 再除以batch_size 就得到均值
                end_points['sp_fusion_logits'] = end_points['sp_fusion_logits'].sum(dim=0) / FLAGS.batch_size
                end_points['sp_logits'] = end_points['sp_logits'].sum(dim=0) / FLAGS.batch_size

                end_points['sp_fusion_logits'] = F.softmax(end_points['sp_fusion_logits'].transpose(0, 1), dim=1)
                end_points['sp_logits'] = F.softmax(end_points['sp_logits'].transpose(0, 1), dim=1)

                # save prediction
                if self.test_dataset.data_list[cloud_inds[0]][0] not in self.saveed_pred_dir:
                    self.saveed_pred_dir.append(self.test_dataset.data_list[0][0])
                    root_dir = os.path.join(FLAGS.result_dir, self.test_dataset.data_list[cloud_inds[0]][0], 'predictions')
                    os.makedirs(root_dir, exist_ok=True)
                    # fusion
                    fu_dir = os.path.join(FLAGS.result_dir, self.test_dataset.data_list[cloud_inds[0]][0], 'fu_prob')
                    os.makedirs(fu_dir, exist_ok=True)
                   
                    vo_dir = os.path.join(FLAGS.result_dir, self.test_dataset.data_list[cloud_inds[0]][0], 'vo_prob')
                    os.makedirs(vo_dir, exist_ok=True)

                pred = np.argmax(end_points['sp_fusion_logits'].cpu().numpy(), 1).astype(np.uint32)
                # pred += 1
                if FLAGS.index_to_label is True:  # 0 - 259
                    # pred = self.remap(pred)
                    name = self.test_dataset.data_list[cloud_inds[0]][1] + '.label'
                    output_path = os.path.join(root_dir, name)
                    pred.tofile(output_path)
                else:  # 0 - 19
                    name = self.test_dataset.data_list[cloud_inds[0]][1] + '.npy'
                    output_path = os.path.join(root_dir, name)
                    np.save(output_path, pred)
                if FLAGS.only_pred == 0:
                   
                    # pc_pro = self.pc_probs[j].astype(np.float32) / count_
                    output_path_pro = os.path.join(fu_dir,
                                                name.replace('.label',
                                                                '_fu.npy') if 'label' in name else name.replace(
                                                    '.npy', '_fu.npy'))
                    np.save(output_path_pro, end_points['sp_fusion_logits'].cpu().numpy())

                    # vo_pro = self.voxel_probs[j].astype(np.float32) / count_
                    output_path_pro = os.path.join(vo_dir,
                                                name.replace('.label',
                                                                '_sp.npy') if 'label' in name else name.replace(
                                                    '.npy', '_sp.npy'))
                    np.save(output_path_pro, end_points['sp_logits'].cpu().numpy())


def main():
    tester = Tester()
    tester.test()


if __name__ == '__main__':
    main()


