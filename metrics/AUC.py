# -*- coding = utf-8 -*-
# @Time: 2025/2/17 20:51
# @Author: wisehone
# @File: AUC.py
# @SoftWare: PyCharm

import numpy as np
from sklearn import metrics

from utils.eval_utils import range_convers_new


def extend_positive_label(x, window=16):
    label = x.copy().astype(np.float32)
    L = range_convers_new(label)
    length = len(label)
    for k in range(len(L)):
        s, e = L[k][0], L[k][1]
        x1 = np.arange(e, min(e + window // 2, length))
        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / window)

    label = np.minimum(np.ones(length), label)
    return label

def extend_postive_range(x, window=16):
    label = x.copy().astype(np.float32)
    L = range_convers_new(label)
    length = len(label)
    for k in range((len(L))):
        s, e = L[k][0], L[k][1]
        x1 = np.arange(e, min(e + window // 2, length))
        label[x1] += np.sqrt(1 - (x1 - e) / window)
        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / window)

    label = np.minimum(np.ones(length), label)
    return label

def extend_postive_range_individual(x, percentage=0.2):
    label = x.copy().astype(np.float32)
    L = range_convers_new(label)
    length = len(label)
    for k in range((len(L))):
        s, e = L[k][0], L[k][1]
        l0 = int((e - s + 1) * percentage)

        x1 = np.arange(e, min(e + l0, length))
        label[x1] += np.sqrt(1 - (x1 - e) / (2 * l0))

        x2 = np.arange(max(s - l0, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / (2 * l0))
    label = np.minimum(np.ones(length), label)
    return label

def TPR_FPR_RangeAUC(labels, pred, P, L):
    product = labels * pred
    TP = np.sum(product)
    P_new = (P + np.sum(labels)) / 2

    recall = min(TP / P_new, 1)

    existence = 0
    for seg in L:
        if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:
            existence += 1

    existence_ratio = existence / len(L)
    TPP_RangeAUC = recall * existence_ratio
    FP = np.sum(pred) - TP

    N_new = len(labels) - P_new
    FPR_RangeAUC = FP / N_new

    Precision_RangeAUC = TP / np.sum(pred)

    return TPP_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

def Range_AUC(score_t_test, y_test, window=5, percentage=0., plot_ROC=False, AUC_type='window'):
    # AUC_type='window'/'percentage'
    score = score_t_test
    label = y_test
    score_sorted = -np.sort(-score)

    P = np.sum(label)
    if AUC_type == 'window':
        label = extend_positive_label(label, window=window)
    else:
        label = extend_postive_range_individual(label, percentage=percentage)

    L = range_convers_new(label)
    TPR_list = [0]
    FPR_list = [0]
    Precision_list = [1]

    for i in np.linspace(0, len(score) - 1, 250).astype(np.int32):
        threshold = score_sorted[i]
        pred = score >= threshold
        TPR, FPR, Precision = TPR_FPR_RangeAUC(label, pred, P, L)

        TPR_list.append(TPR)
        FPR_list.append(FPR)
        Precision_list.append(Precision)

    TPR_list.append(1)
    FPR_list.append(1)

    tpr = np.array(TPR_list)
    fpr = np.array(FPR_list)
    prec = np.array(Precision_list)

    width = fpr[1:] - fpr[:-1]
    height = (tpr[1:] + tpr[:-1]) / 2
    AUC_range = np.sum(height * width)

    width_PR = tpr[1: -1] - tpr[:-2]
    height_PR = (prec[1:] + prec[:-1]) / 2
    AP_range = np.sum(height_PR * width_PR)

    if plot_ROC:
        return AUC_range, AP_range, tpr, fpr, prec
    return AUC_range

def point_wise_AUC(score_t_test, y_test, plot_ROC=False):
    label = y_test
    score = score_t_test
    auc = metrics.roc_auc_score(label, score)
    if plot_ROC:
        fpr, tpr, thresholds = metrics.roc_curve(label, score)
        return auc, fpr, tpr, thresholds
    return auc

if __name__ == '__main__':
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:17] = 0.5
    pred_labels[55:62] = 0.7
    # pred_labels[51:55] = 1
    # true_events = get_events(y_test)
    point_auc = point_wise_AUC(pred_labels, y_test)
    range_auc = Range_AUC(pred_labels, y_test)
    print("point_auc: {}, range_auc: {}".format(point_auc, range_auc))

