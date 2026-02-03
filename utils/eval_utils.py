# -*- coding = utf-8 -*-
# @Time: 2025/2/18 9:12
# @Author: wisehone
# @File: eval_utils.py
# @SoftWare: PyCharm

import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema


def get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t, return_prec_rec=False):
    pred_labels = score_t_test > thres
    tp = np.sum([pred_labels[start:end+1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

class NptConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def find_length(data):
    if len(data.shape) > 1:
        return 0
    data = data[:min(20000, len(data))]
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125

def range_convers_new(label):
    L, i, j = [], 0, 0
    while j < len(label):
        while label[i] == 0:
            i += 1
            if i >= len(label):
                break
        j = i + 1
        if j >= len(label):
            if j == len(label):
                L.append((i, j + 1))
            break
        while label[j] != 0:
            j += 1
            if j >= len(label):
                L.append((i, j - 1))
                break
        if j >= len(label):
            break
        L.append((i, j - 1))
        i = j
    return L

def score_to_label(anomaly_score, threshold):
    anomaly_label = anomaly_score > threshold
    anomaly_label = anomaly_label.astype(np.float32)
    return anomaly_label

"""
def get_npsr_label(true_label, test_score):
    ones = true_label.sum()
    zeros = len(true_label) - ones

    sortid = np.argsort(test_score - true_label * 1e-4)
    new_lab = true_label[sortid]
    new_scores = test_score[sortid]


    TPs = np.cumsum(-new_lab) + ones
        
    FPs = np.cumsum(new_lab-1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs
    
    N = len(true_label) - np.flip(TPs > 1e-5).argmax()
    TPRs = TPs[:N] / ones
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s  = 2 * TPRs * PPVs / (TPRs + PPVs)
    maxid = np.argmax(F1s)
    
    FPRs = np.insert(FPRs, -1, 0)
    TPRs = np.insert(TPRs, -1, 0)

    AUC = roc_auc_score(true_label, test_score)
   
    anomaly_ratio = ones / len(true_label) 
    FPR_bestF1_TPR1 = anomaly_ratio / (1-anomaly_ratio) * (2 / F1s[maxid] - 2)
    TPR_bestF1_FPR0 = F1s[maxid] / (2 - F1s[maxid])

    thres = new_scores[maxid]
    pred = (test_score > thres).astype(int)
    # res = get_all_evaluation_score(pred, true_label)
    # print(res)

    print('--------')
    # print({'AUC': AUC, 'F1': F1s[maxid], 'thres': new_scores[maxid], 'TPR': TPRs[maxid], 'PPV': PPVs[maxid], 
    #         'FPR': FPRs[maxid], 'maxid': maxid, 'FPRs': FPRs, 'TPRs': TPRs, 
    #         'FPR_bestF1_TPR1': FPR_bestF1_TPR1, 'TPR_bestF1_FPR0': TPR_bestF1_FPR0})
    print({'AUC': AUC, 'F1': F1s[maxid], 'thres': new_scores[maxid], 'TPR': TPRs[maxid], 'PPV': PPVs[maxid],
            'FPR_bestF1_TPR1': FPR_bestF1_TPR1, 'TPR_bestF1_FPR0': TPR_bestF1_FPR0})

    return pred
"""
def get_npsr_label(true_label, test_score):
    ones = true_label.sum()
    zeros = len(true_label) - ones

    sortid = np.argsort(test_score - true_label * 1e-2)
    new_lab = true_label[sortid]
    new_scores = test_score[sortid]

    TPs = np.cumsum(-new_lab) + ones
    FPs = np.cumsum(new_lab-1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs
    
    N = len(true_label) - np.flip(TPs > 1e-3).argmax()
    TPRs = TPs[:N] / ones
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s = 2 * TPRs * PPVs / (TPRs + PPVs)
    
    # 找第二大F1值
    maxid1 = np.argmax(F1s)  # 第一大的索引
    
    # 将第一大的值设为负无穷，然后找第二大
    F1s_copy = F1s.copy()
    F1s_copy[maxid1] = -np.inf
    second_maxid = np.argmax(F1s_copy)  # 第二大的索引
    
    FPRs = np.insert(FPRs, -1, 0)
    TPRs = np.insert(TPRs, -1, 0)

    AUC = roc_auc_score(true_label, test_score)
   
    anomaly_ratio = ones / len(true_label) 
    # 使用第二大F1值计算
    FPR_bestF1_TPR1 = anomaly_ratio / (1-anomaly_ratio) * (2 / F1s[second_maxid] - 2)
    TPR_bestF1_FPR0 = F1s[second_maxid] / (2 - F1s[second_maxid])

    thres = new_scores[second_maxid]
    pred = (test_score > thres).astype(int)

    print('--------')
    print({'AUC': AUC, 
           'F1': F1s[second_maxid], 
           'thres': new_scores[second_maxid], 
           'TPR': TPRs[second_maxid], 
           'PPV': PPVs[second_maxid],
           'FPR_bestF1_TPR1': FPR_bestF1_TPR1, 
           'TPR_bestF1_FPR0': TPR_bestF1_FPR0,
           'first_max_F1': F1s[maxid1],  # 可选：显示第一大的F1值作为参考
           'second_max_index': second_maxid})

    return pred