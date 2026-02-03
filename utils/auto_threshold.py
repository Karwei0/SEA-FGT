# -*- coding = utf-8 -*-
# @Time: 2025/10/24 15:46
# @Author: wisehone
# @File: auto_threshold.py
# @SoftWare: PyCharm
from typing import Iterable, Dict, Optional, Tuple
from metrics.get_target_metric import *
import numpy as np

# the original f1/P/R
def _f1_from_labels(pred: np.ndarray, true: np.ndarray, eps: float=1e-12):
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def grid_search_percentile_k(
        val_scores_1d: np.ndarray,
        val_labels_1d: np.ndarray,
        k_candidates: Iterable[float],
        train_scores_1d: Optional[np.ndarray] = None
) -> Tuple[float, float, Dict[float, Dict[str, float]]]:
    """
    Try to find the best k% percentile threshold.
    - If `train_scores_1d` is provided, thresholds are computed from it,
      then applied to val_scores_1d for evaluation.
    - Otherwise, use val_scores_1d itself for both percentile and evaluation.

    Args:
        val_scores_1d: 1D array of validation scores.
        val_labels_1d: 1D array of validation labels.
        k_candidates: list/iterable of percentile candidates (e.g., [90, 95, 97]).
        train_scores_1d: (Optional) 1D array of training scores (for percentile reference).

    Returns:
        best_k: percentile (float)
        best_threshold: threshold value (float)
        all_stat: dict of stats per candidate {k: {'threshold', 'precision', 'recall', 'f1'}}
    """
    assert val_scores_1d.ndim == 1 and val_labels_1d.ndim == 1
    assert val_scores_1d.shape[0] == val_labels_1d.shape[0] 

    ks = list(sorted(set(float(k) for k in k_candidates)))
    all_stat: Dict[float, Dict[str, float]] = {}
    best_f1, best_k, best_th = -1.0, None, None

    # choose the reference distribution for percentile
    ref_scores = train_scores_1d if train_scores_1d is not None else val_scores_1d

    for k in ks:
        q = 100.0 - float(k)

        # use train scores for percentile if available
        th = float(np.percentile(ref_scores, q))
        pred = (val_scores_1d > th).astype(np.int32)
        p, r, f1 = _f1_from_labels(pred, val_labels_1d)
        # f1_pa = get_target_metric('pa_f1', pred, val_labels_1d)

        all_stat[k] = {'threshold': th, 'precision': p, 'recall': r, 'f1': f1}
        # print(f'all_stat[{k}]: ', all_stat[k])
        if f1 > best_f1:
            best_f1, best_k, best_th = f1, k, th

    return best_k, best_th, all_stat