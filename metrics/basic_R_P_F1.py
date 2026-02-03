# -*- coding = utf-8 -*-
# @Time: 2025/2/17 15:47
# @Author: wisehone
# @File: basic_R_P_F1.py
# @SoftWare: PyCharm
import argparse, os

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.eval_utils import score_to_label

def cal_R_P_F1(true_anomaly_label, predict_anomaly_label, threshold=0.5):
    precision = precision_score(true_anomaly_label, predict_anomaly_label)
    recall = recall_score(true_anomaly_label, predict_anomaly_label)
    f1 = f1_score(true_anomaly_label, predict_anomaly_label)
    return precision, recall, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='SMD', help='dataset name')
    parser.add_argument('--model', type=str, default='DAT', help='model name')

    parser.add_argument('--predict_anomaly_score', type=str, required=True, help='model predicts anomaly score file(.npy)')
    parser.add_argument('--true_anomaly_label', type=str, required=True, help='true anomaly label file(.npy)')

    parser.add_argument('--thresholds', type=(float, list), default=[0.5], help='the threshold to detect anomaly')

    options = parser.parse_args()

    print('dataset: {}, model: {}'.format(options.dataset, options.model))
    if not os.path.exists(os.path.join('result', options.model, options.dataset)):
        os.makedirs(os.path.join('result', options.model, options.dataset))

    with open(os.path.join('result', options.model, options.dataset, 'basic_R_P_F1.txt'), 'w') as f:
        # thresholds, precision, recall, f1
        for threshold in options.thresholds:
            precision, recall, f1 = cal_R_P_F1(options.predict_anomaly_score, options.true_anomaly_label, threshold)
            print(
                'threshold: {}, precision: {}, recall: {}, f1: {}'.format(options.dataset, threshold, precision, recall,
                                                                          f1))
            f.write(f'{threshold},{precision},{recall},{f1}\n')









