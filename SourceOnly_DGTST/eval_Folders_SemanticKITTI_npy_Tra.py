#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from utils.np_ioueval import iouEval
import pickle
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser("./evaluate_semantics.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='~/dataset/semanticKITTI/dataset/sequences',
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default='res_pred/Ori_M34_XYZ_Sp',
        help='Prediction dir. Same organization as dataset, but predictions in'
        'each sequences "prediction" directory. No Default. If no option is set'
        ' we look for the labels in the same directory as dataset'
    )
    parser.add_argument(
        '--sequences',  # '-l',
        # ['00', '01', '02', '03','04', '05', '06', '07', '09', '10']   ['08'],
        nargs="+",
        default= ['08'] ,
        help='evaluated sequences',
    )

    parser.add_argument(
        '--datacfg', '-dc',
        type=str,
        required=False,
        default="utils/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
        ' evaluating single scan from aggregated pointcloud.'
        ' Defaults to %(default)s',
    )
    FLAGS = parser.parse_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset
    return FLAGS


def load_label(data_root, sequences, sub_dir_name, ext):
    label_names = []
    for sequence in sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(data_root, str(sequence), sub_dir_name)
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if f".{ext}" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)
    return label_names


if __name__ == '__main__':

    FLAGS = get_args()
    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Sequences: ", FLAGS.sequences)
    print("Config: ", FLAGS.datacfg)
    print("Limit: ", FLAGS.limit)
    print("*" * 80)

    print("Opening data config file %s" % FLAGS.datacfg)
    DATA = yaml.safe_load(open(FLAGS.datacfg, 'r'))

    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_ignore = DATA["learning_ignore"]
    # class_inv_remap = DATA["learning_map_inv"]
    nr_classes = 20  # len(class_inv_remap)

    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("Ignoring xentropy class ", x_cl, " in IoU evaluation")
    # create evaluator
    evaluator = iouEval(nr_classes, ignore)
    evaluator.reset()

    # get label paths
    dataset_path = os.path.expanduser(FLAGS.dataset)
    label_names = load_label(FLAGS.dataset, FLAGS.sequences, "labels", "label")

    pred_names = load_label(
        FLAGS.predictions, FLAGS.sequences, "predictions", "npy")

    assert(len(label_names) == len(pred_names))

    print("Evaluating sequences")
    N = len(label_names)
    # open each file, get the tensor, and make the iou comparison
    for i in tqdm(range(N), ncols=50):
        label_file = label_names[i]
        pred_file = pred_names[i]
        # open label

        label = np.fromfile(label_file, dtype=np.int32)
        label = label.reshape((-1)).astype(np.int32)
        # label = np.load(label_file).astype(np.int32)
        label = label & 0xFFFF  # semantic label in lower half
        label = remap_lut[label]

        # label = label & 0xFFFF
        if FLAGS.limit is not None:
            label = label[:FLAGS.limit]  # limit to desired length
        # open prediction
        # pred = np.load(pred_file)
        pred = np.load(pred_file).astype(np.int32)
        pred = pred.reshape((-1))    # reshape to vector

        if FLAGS.limit is not None:
            pred = pred[:FLAGS.limit]  # limit to desired length
        # add single scan to evaluation
        evaluator.addBatch(pred, label)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    print('Validation set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy, m_jaccard=m_jaccard))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                  i=i, class_str=class_strings[i], jacc=jacc))

    # print for spreadsheet
    print("*" * 80)
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()
