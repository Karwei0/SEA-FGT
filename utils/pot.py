# -*- coding = utf-8 -*-
# @Time: 2025/3/2 20:14
# @Author: wisehone
# @File: pot.py
# @SoftWare: PyCharm
import os

import numpy as np
from utils.spot import SPOT
# from src.constants import lm  # Assume lm = [0.02, 3]
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from metrics.get_all_evaluation_score import get_all_evaluation_score

class POTThresholdDetector:
    def __init__(self, q=1e-5, level=0.02, scale_factor=3.0):
        """
        Dynamic threshold detector based on Extreme Value Theory (SPOT)

        Args:
            q (float): Risk parameter, controls sensitivity (default 1e-5)
            level (float): Initial threshold quantile (default 0.02)
            scale_factor (float): Final threshold scaling factor (default 3.0)
        """
        self.q = q
        self.level = level
        self.scale_factor = scale_factor
        self.spot = None
        self.threshold = None

    def fit(self, train_scores, test_scores):
        """
        Train the model and compute dynamic threshold

        Args:
            train_scores (np.ndarray): Training set anomaly scores (for SPOT initialization)
            test_scores (np.ndarray): Test set anomaly scores (for detection)
        """
        # Dynamically adjust initial quantile until successful
        current_level = self.level
        while True:
            try:
                self.spot = SPOT(self.q)
                self.spot.fit(train_scores, test_scores)
                self.spot.initialize(level=current_level, min_extrema=False, verbose=False)
                break
            except:
                current_level *= 0.98  # Gradually reduce the quantile

        # Run SPOT to get thresholds
        results = self.spot.run(dynamic=False)

        # Compute final threshold
        if len(results['thresholds']) > 0:
            self.threshold = np.mean(results['thresholds']) * self.scale_factor
        else:
            self.threshold = np.percentile(test_scores, 100 * (1 - self.level)) * self.scale_factor

    def predict(self, scores):
        """
        Generate prediction labels based on threshold

        Args:
            scores (np.ndarray): Anomaly scores to predict

        Returns:
            np.ndarray: Binarized prediction labels (0/1)
        """
        return (scores > self.threshold).astype(int)

    def evaluate(self, scores, labels):
        """
        Evaluate threshold effectiveness

        Args:
            scores (np.ndarray): Test set anomaly scores
            labels (np.ndarray): Ground truth labels (0/1)

        Returns:
            dict: Dictionary containing F1, Precision, Recall, AUC and other metrics
        """
        pred = self.predict(scores)
        res = get_all_evaluation_score(pred, labels)

        res = {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in res.items()
        }

        return res, self.threshold