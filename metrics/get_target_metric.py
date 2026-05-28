import numpy as np

from metrics.AUC import point_wise_AUC, Range_AUC
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.F1_pa import *
from metrics.vus.metrics import get_range_vus_roc
from utils.eval_utils import score_to_label
def get_target_metric(target_metric, pred_labels, labels_1d):
     # ['pa_f1', 'pa_rec', 'pa_pre', 'aff_rec', 'aff_pre', 'aff_f1', 'f1', 'f05', 'rec', 'prec']
     """
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
     """
     if target_metric in ['aff_rec', 'aff_pre', 'aff_f1']:
          events_pred = convert_vector_to_events(pred_labels)
          events_gt = convert_vector_to_events(labels_1d)
          Trange = (0, len(labels_1d))
          affiliation = pr_from_events(events_pred, events_gt, Trange)
          if target_metric == 'aff_rec':
              return affiliation['recall']
          elif target_metric == 'aff_pre':
              return affiliation['precision']
          elif target_metric == 'aff_f1':
              return affiliation['f1']
     elif target_metric == 'f1':
          _, _, _, f1_score_ori, _ = get_accuracy_precision_recall_f1(pred_labels, labels_1d)
          return f1_score_ori
     elif target_metric == 'f05':
          _, _, _, _, f05_score_ori = get_accuracy_precision_recall_f1(pred_labels, labels_1d)
          return f05_score_ori
     elif target_metric == 'prec':
          _, precision_ori, _, _, _ = get_accuracy_precision_recall_f1(pred_labels, labels_1d)
          return precision_ori
     elif target_metric == 'rec':
          _, _, recall_ori, _, _ = get_accuracy_precision_recall_f1(pred_labels, labels_1d)
          return recall_ori
     elif target_metric == 'pa_f1':
          true_events = get_events(labels_1d)
          _, _, _, _, _, f1_score_pa = get_point_adjust_f1_score(pred_labels, labels_1d, true_events)
          return f1_score_pa
     elif target_metric == 'pa_pre':
          true_events = get_events(labels_1d)
          _, _, _, precision_pa, _, _ = get_point_adjust_f1_score(pred_labels, labels_1d, true_events)
          return precision_pa
     elif target_metric == 'pa_rec':
          true_events = get_events(labels_1d)
          _, _, _, _, recall_pa, _ = get_point_adjust_f1_score(pred_labels, labels_1d, true_events)
          return recall_pa
     elif target_metric == 'point_auc':
          point_auc = point_wise_AUC(pred_labels, labels_1d)
          return point_auc
     elif target_metric == 'range_auc':
          range_auc = Range_AUC(pred_labels, labels_1d)
          return range_auc
     elif target_metric == 'vus_roc':
          results = get_range_vus_roc(labels_1d, pred_labels, 100)  # slidingWindow = 100 default
          return results["VUS_ROC"]
     elif target_metric == 'vus_pr':
          results = get_range_vus_roc(labels_1d, pred_labels, 100)  # slidingWindow = 100 default
          return results["VUS_PR"]
     elif target_metric == 'r_auc_roc':
          results = get_range_vus_roc(labels_1d, pred_labels, 100)  # slidingWindow = 100 default
          return results["R_AUC_ROC"]
     elif target_metric == 'r_auc_pr':
          results = get_range_vus_roc(labels_1d, pred_labels, 100)  # slidingWindow = 100 default
          return results["R_AUC_PR"]
     return 0