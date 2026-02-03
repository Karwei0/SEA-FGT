# -*- coding = utf-8 -*-
# @Time: 2025/2/17 20:26
# @Author: wisehone
# @File: get_all_evaluation_score.py
# @SoftWare: PyCharm
import numpy as np

from metrics.AUC import point_wise_AUC, Range_AUC
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.F1_pa import *
from metrics.vus.metrics import get_range_vus_roc
from utils.eval_utils import score_to_label



def get_all_evaluation_score(predictive_label, true_label):
    # get affiliation indicator
    events_pred = convert_vector_to_events(predictive_label)  # [(4, 5), (8, 9)]
    events_gt = convert_vector_to_events(true_label)  # [(3, 4), (7, 10)]
    Trange = (0, len(true_label))
    affiliation = pr_from_events(events_pred, events_gt, Trange)

    # get basic/original recall, precision, f1_score
    _, precision_ori, recall_ori, f1_score_ori, f05_score_ori = get_accuracy_precision_recall_f1(predictive_label, true_label)

    # another way to get basic recall, precision, f1_score

    # get basic recall, precision, f1_score with PA
    true_events = get_events(true_label)
    _, _, _, precision_pa, recall_pa, f1_score_pa = get_point_adjust_f1_score(predictive_label, true_label, true_events)

    # get point auc and range auc
    point_auc = point_wise_AUC(predictive_label, true_label)
    range_auc = Range_AUC(predictive_label, true_label)

    # get about VUS indicators (V_ROC and V_RR)
    results = get_range_vus_roc(true_label, predictive_label, 100)  # slidingWindow = 100 default

    score_list = {
                'recall_ori': recall_ori,
                'precision_ori': precision_ori,
                "f1_score_ori": f1_score_ori,
                "f05_score_ori": f05_score_ori,

                "precision_pa": precision_pa,
                "recall_pa": recall_pa,
                "f_score_pa": f1_score_pa,

                "point_auc": point_auc,
                "range_auc": range_auc,
                "Affiliation precision": affiliation['precision'],
                "Affiliation recall": affiliation['recall'],

                "R_AUC_ROC": results["R_AUC_ROC"],
                "R_AUC_PR": results["R_AUC_PR"],
                "VUS_ROC": results["VUS_ROC"],
                "VUS_PR": results["VUS_PR"]
    }

    return score_list

if __name__ == '__main__':
    true_label = np.zeros(100)
    true_label[10:20] = 1
    true_label[55:57] = 1
    true_label[80:90] = 1

    anomaly_score = np.zeros(100)
    anomaly_score[15:17] = 0.5
    anomaly_score[55:62] = 0.7
    anomaly_score[51:55] = 1
    anomaly_score[80:90] = 1

    predictive_label = score_to_label(anomaly_score, 0.65)
    print(get_all_evaluation_score(predictive_label, true_label))
