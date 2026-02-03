# -*- coding = utf-8 -*-
# @Time: 2025/3/2 20:14
# @Author: wisehone
# @File: pot.py
# @SoftWare: PyCharm
import os

import numpy as np
from utils.spot import SPOT
# from src.constants import lm  # 假设 lm = [0.02, 3]
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from metrics.get_all_evaluation_score import get_all_evaluation_score

class POTThresholdDetector:
    def __init__(self, q=1e-5, level=0.02, scale_factor=3.0):
        """
        基于极值理论 (SPOT) 的动态阈值检测器

        Args:
            q (float): 风险参数，控制灵敏度 (默认 1e-5)
            level (float): 初始阈值分位数 (默认 0.02)
            scale_factor (float): 最终阈值的缩放因子 (默认 3.0)
        """
        self.q = q
        self.level = level
        self.scale_factor = scale_factor
        self.spot = None
        self.threshold = None

    def fit(self, train_scores, test_scores):
        """
        训练模型并计算动态阈值

        Args:
            train_scores (np.ndarray): 训练集异常分数 (用于初始化 SPOT)
            test_scores (np.ndarray): 测试集异常分数 (用于检测)
        """
        # 动态调整初始分位数直到成功
        current_level = self.level
        while True:
            try:
                self.spot = SPOT(self.q)
                self.spot.fit(train_scores, test_scores)
                self.spot.initialize(level=current_level, min_extrema=False, verbose=False)
                break
            except:
                current_level *= 0.98  # 逐步降低分位数

        # 运行 SPOT 获取阈值
        results = self.spot.run(dynamic=False)

        # 计算最终阈值
        if len(results['thresholds']) > 0:
            self.threshold = np.mean(results['thresholds']) * self.scale_factor
        else:
            self.threshold = np.percentile(test_scores, 100 * (1 - self.level)) * self.scale_factor

    def predict(self, scores):
        """
        根据阈值生成预测标签

        Args:
            scores (np.ndarray): 需要预测的异常分数

        Returns:
            np.ndarray: 二值化预测标签 (0/1)
        """
        return (scores > self.threshold).astype(int)

    def evaluate(self, scores, labels):
        """
        评估阈值效果

        Args:
            scores (np.ndarray): 测试集异常分数
            labels (np.ndarray): 真实标签 (0/1)

        Returns:
            dict: 包含 F1、Precision、Recall、AUC 等指标
        """
        pred = self.predict(scores)
        res = get_all_evaluation_score(pred, labels)

        res = {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in res.items()
        }

        return res, self.threshold

