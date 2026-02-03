# -*- coding = utf-8 -*-
# @Time: 2025/2/17 20:29
# @Author: wisehone
# @File: F1_pa.py
# @SoftWare: PyCharm

import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, fbeta_score


def get_point_adjust_f1_score(pred_y, true_y, true_events, threshold_k=0, wheather_top_k=False):
    """

    :param pred_y: predict labels (0/1)
    :param true_y: true labels
    :param true_events:
    :param threshold_k:
    :param wheather_top_k:
    :return:
    """
    tp, fn = 0., 0.
    for true_event, (true_start, true_end) in true_events.items():
        if not wheather_top_k:
            if pred_y[true_start:true_end].sum() > 0:
                tp += (true_end - true_start)
            else:
                fn += (true_end - true_start)
        else:
            if pred_y[true_start:true_end].sum() > threshold_k:
                tp += (true_end - true_start)
            else:
                fn += (true_end - true_start)
    fp = np.sum(pred_y) - np.sum(pred_y * true_y)

    prec, rec, fscore = get_prec_rec_fscore(tp, fp, fn)
    return fp, fn, tp, prec, rec, fscore

def get_adjust_f1_PA(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    accuracy = accuracy_score(gt, pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(gt, pred, average='binary')

    return accuracy, precision, recall, fscore

def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore

def get_f_score(precision, recall):
    if precision == 0 and recall == 0:
        return 0

    f_score = 2 * precision * recall / (precision + recall)
    return f_score

def get_accuracy_precision_recall_f1(pred, gt):
    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    if precision == 0 and recall == 0:
        f_score = 0
    else:
        f_score = (2 * precision * recall) / (precision + recall)
    if precision == 0 and recall == 0:
        f05_score = 0
    else:
        f05_score = fbeta_score(gt, pred, beta=0.5)
    return accuracy, precision, recall, f_score, f05_score

def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events
