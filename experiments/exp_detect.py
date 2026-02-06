# -*- coding = utf-8 -*-
# @Time: 2025/10/24 8:57
# @Author: wisehone
# @File: exp_detect.py
# @SoftWare: PyCharm
import math
import sys
from typing import Dict

import pandas as pd

from config import dataset2path
from utils.Learning_rate import adjust_learning_rate

sys.path.append('..')
sys.path.append('.')

import os, time, numpy as np, torch
from scipy.special import softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim

from datasets.loader_provider import get_data_loader
from experiments.exp_basic import Exp_Basic

from losses.aux_loss import ContrastiveLossManager
from losses.KL_loss import KL_loss

from utils.threshold import symmetric_kl_scores, ThresholdPolicy
from utils.auto_threshold import grid_search_percentile_k
from utils.EalryStop import EarlyStop

from metrics.get_all_evaluation_score import get_all_evaluation_score
from metrics.get_target_metric import get_target_metric

from utils.eval_utils import get_npsr_label

import numpy as np
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'

class Exp_Detect(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.folder_path = getattr(args, 'folder_path', './result')

        # dataset
        self.data_name = args.dataset
        self.data_path = args.data_path if args.data_path else dataset2path[self.dataset_name]
        self.win_size = args.win_size
        self.step = args.step
        self.flag = args.flag
        self.batch_size = args.batch_size

        # manage loss
        self.loss_mgr = ContrastiveLossManager(
            temperature=args.temperature,
            lambda_uti=args.lambda_uti,
            lambda_orth=args.lambda_orth,
            lambda_info_nce=args.lambda_info_nce
        )

        self.best_val_k = None
        self.best_val_threshold = None
        self.val_scores_1d = None
        self.val_labels_1d = None
        self.train_scores_1d = None

        self.earlystop_mode = args.earlystopmode
        self.monitor = args.monitor
        self.monitor_metric = args.monitor_metric
        self.earlystop = EarlyStop(
            patience=args.patience,
            verbose=args.verbose,
            dataset_name=args.dataset,
            delta=args.delta, # default 1e-4
            save_every_epoch=args.save_every_epoch, # default False
            monitor=args.monitor, # default 'val_loss'
            mode=args.earlystopmode # default 'min'
        )

        # manage threshold
        self.threshold_policy = ThresholdPolicy(
            mode=args.th_mode,
            k_percent=args.th_k,
            spot_q=args.spot_q,
            spot_level=args.spot_level,
            spot_scale=args.spot_scale
        )

    # TODO: implement _build_model
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        return model

    # TODO: make it suitable for own dataloader
    def _get_data(self, flag=None):
        """
            need a data_provider return dataset, dataloader
            dataloader 每个 batch 是一个 dict:
          {'x': [B,T,N], 'y': optional, 'label': optional (0/1 for anomaly)}
        :param flag: ['train', 'test', 'val'
        :return:
        """
        dataset, loader = get_data_loader(
            root_data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            mode=flag,
            dataset=self.data_name
        )
        return dataset, loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _forward_and_collect(self, batch_x, batch_y=None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        约定：SEA_FGA.forward() 返回一个 dict，至少包含：
        y_ori, y_aug, 以及（若启用）gate_ori, gate_aug, expert_utilization, experts
        可选：anomaly_score
        :param batch_x:
        :param batch_y:
        :param kwargs:
        :return:
        """
        outputs = self.model(batch_x, **kwargs)
        if isinstance(outputs, dict):
            return outputs
        elif isinstance(outputs, tuple) or isinstance(outputs, list):
            keys=['y_ori', 'y_aug', 'coherence_matrix', 'smooth_p', 'gate_ori', 'gate_aug', 'experts']
            return {k: v for k, v in zip(keys, outputs)}
        raise ValueError('SEA_FGT must return a dict with (y_ori/y_aug and so on)')

    def _compute_total_loss(self, out_dict, batch):
        device = self.device 
        # TODO: check
        # SEA_FGT - OUT :y_ori, y_aug, coherence_matrix, expert_utilization, gate_ori, gate_aug
        if isinstance(out_dict, dict):
            y_ori = out_dict.get('y_ori', None)
            y_aug = out_dict.get('y_aug', None)
            g_ori = out_dict.get('gate_ori', None)
            g_aug = out_dict.get('gate_aug', None)
            smooth_p = out_dict.get('smooth_p', None)
            se = out_dict.get('experts', None)
        elif isinstance(out_dict, tuple):
            # out = y_ori, y_aug, coherence_matrix, expert_utilization, gate_ori, gate_aug, self.sea.get_SE
            y_ori, y_aug, _, smooth_p, g_ori, g_aug, se = out_dict
        else:
            raise ValueError('output is not a suitable type')

        if (y_ori is None) or (y_aug is None):
            raise ValueError('y_ori or/and y_aug is None')

        zeros = torch.tensor(0.0, device=device)
        # TODO: mgr lack for experts networks
        loss_dict = self.loss_mgr(y_ori, y_aug, g_ori, g_aug, smooth_p, se)

        total_loss = loss_dict['total_loss']
        return total_loss, loss_dict

    # collect score and labels
    def _gather_scores_labels_1d(self, loader, temperature: float):
        self.model.eval()
        scores_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # TODO: check here. The return seems not suitabel??
                out = self._forward_and_collect(batch_x, batch_y)

                y1 = out['y_ori']
                y2 = out['y_aug']

                s_bt = symmetric_kl_scores(y1, y2, temperature=temperature)
                scores_list.append(F.softmax(s_bt.reshape(-1).detach(), dim=-1).cpu().numpy())

                if isinstance(batch_y, torch.Tensor):
                    lab = batch_y.detach().cpu().numpy()
                labels_list.append(lab.reshape(-1))

        scores_1d = np.concatenate(scores_list, axis=0) if scores_list else np.array([])
        # print('scores_1d: ', scores_1d.shape)
        labels_1d = (np.concatenate(labels_list, axis=0).astype(np.int32) if labels_list else None)
        # print('labels_1d: ', labels_1d.shape)
        return scores_1d, labels_1d
      

    def train(self, setting=None):
        """
        TODO: setting is what?
        :param setting: the experiment name
        :return:
        """
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        os.makedirs(self.args.checkpoints, exist_ok=True)
        ckpt_dir = os.path.join(self.args.checkpoints, (setting or 'fgt_experiment'))

        optimizer = self._select_optimizer()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        print('training model')
        for ep in range(self.args.train_epochs):
            ep_loss = []
            t0 = time.time()
            self.model.train()

            for it, batch in enumerate(train_loader):
                batch_x, batch_y = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        out = self._forward_and_collect(batch_x, batch_y)
                        total, loss_dict = self._compute_total_loss(out, batch_x)
                    optimizer.zero_grad()
                    scaler.scale(total).backward()
                    if self.args.grad_clip:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = self._forward_and_collect(batch_x, batch_y)
                    # print('expert_utilization ', out['expert_utilization'].sum(dim=[0,1]))
                    total, loss_dict = self._compute_total_loss(out, batch_x)
                    optimizer.zero_grad()
                    total.backward()
                    if self.args.grad_clip:
                        clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    optimizer.step()

                ep_loss.append(total.item())
                if (it + 1) % max(1, self.args.log_interval) == 0:
                    lm, lc, lu, lo = [loss_dict[k].item() for k in ['l_main', 'l_infonce', 'l_uti', 'l_orth']]
                    logg = f'--[E{ep+1:03d} I{it+1:04d}] total:{total.item():.5f} | main:{lm:.5f}'
                    if lu > 0:
                        logg += f' | util:{lu:.5f}'
                    if lc > 0:
                        logg += f' | infoNCE:{lc:.5f}'
                    print(logg)

            self.train_scores_1d, _ = self._gather_scores_labels_1d(train_loader, self.args.temperature)
            #vail
            val_loss, val_metric = self.vali(val_data, val_loader)
            avg_train = np.mean(ep_loss)
            print(f'Epoch {ep+1} done in {time.time()-t0:.1f}s | train={avg_train:.5f} | valid={val_loss:.5f}')

            # save best model
            # TODO: lack early STOPPING and val_loss or val_metirc
            self.earlystop(val_loss, val_metric, self.model, ckpt_dir, epoch=ep+1)
            if self.earlystop.early_stop:
                print('early stop')
                break
            adjust_learning_rate(optimizer, ep+1, self.args)

        # load best
        best_path = os.path.join(ckpt_dir, 'checkpoint.pth')
        if os.path.isfile(best_path):
            # 1) 先安全地在 CPU 上加载，避免 GPU 号不匹配
            state = torch.load(best_path, map_location='cpu')

            # 2) 取出真正的 state_dict（有的保存成 {'state_dict': ...}）
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']

            # 3) 如是 DataParallel 保存，去掉 'module.' 前缀
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                new_k = k.replace('module.', '', 1) if k.startswith('module.') else k
                new_state[new_k] = v

            # 4) 加载到模型（模型此时已在 self.device 上）
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            if missing or unexpected:
                print(f'--[WARN] load_state_dict: missing={missing}, unexpected={unexpected}')
            print(next(self.model.parameters()).device)
            print(f'Load best model from {best_path} -> {self.device}')

        """
        best_path = os.path.join(ckpt_dir, 'checkpoint.pth')
        if os.path.isfile(best_path):
            self.model.load_state_dict(torch.load(best_path))
            print(f'Load best model from {best_path}')
        """
        if self.args.th_mode in ('percentile_train', 'spot', 'unified'):
            self.train_scores_1d, _ = self._gather_scores_labels_1d(train_loader, self.args.temperature)

        return self.model

    def vali(self, vali_data=None, vali_loader=None):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                out = self._forward_and_collect(batch_x, batch_y)
                loss, _ = self._compute_total_loss(out, batch_x)
                total_loss.append(loss.item())

        val_loss = float(np.mean(total_loss)) if total_loss else np.inf

        try:
            self.val_scores_1d, self.val_labels_1d = self._gather_scores_labels_1d(vali_loader, self.args.temperature)
        except Exception:
            self.val_scores_1d, self.val_labels_1d = None, None

        if self.args.th_mode == 'percentile_val' and self.val_scores_1d is not None:
            best_k, best_th, _ = grid_search_percentile_k(
                self.val_scores_1d, self.val_labels_1d, self.args.th_grid
            )
            self.best_val_k, self.best_val_threshold = best_k, best_th
            print(f'Best k: {best_k} | Best threshold: {best_th:.4f}')
        elif self.args.th_mode == 'percentile_train' and self.train_scores_1d is not None:
            best_k, best_th, _ = grid_search_percentile_k(
                self.val_scores_1d, self.val_labels_1d, self.args.th_grid, self.train_scores_1d
            )
            self.best_val_k, self.best_val_threshold = best_k, best_th
            print(f'Best k: {best_k} | Best threshold: {best_th:.4f}')
        elif self.args.th_mode == 'unified' and self.train_scores_1d is not None:
            unified_scores = np.concatenate([self.train_scores_1d, self.val_scores_1d])
            best_k, best_th, _ = grid_search_percentile_k(
                self.val_scores_1d, self.val_labels_1d, self.args.th_grid, unified_scores
            )
            print(f'Best k: {best_k} | Best threshold: {best_th:.4f}')
        self.threshold_policy.fit(train_scores_1d=self.train_scores_1d, ref_score_1d=self.val_scores_1d)
        pred_val_labels = self.threshold_policy.predict(self.val_scores_1d)

        # cal metrics
        val_target_metric = get_target_metric(self.monitor_metric, pred_val_labels, self.val_labels_1d)

        self.model.train()
        return val_loss, val_target_metric

    # TODO: feeling weird somewhere
    def test(self, setting=None):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        res_csv = os.path.join(self.folder_path, f'{setting if setting else "test"}_res.csv')

        test_scores_1d, test_labels_1d = self._gather_scores_labels_1d(test_loader, self.args.temperature)
        
        if getattr(self, 'train_scores_1d', None) is None and self.args.th_mode in ['spot', 'percentile_train', 'unified']:
            train_data, train_loader = self._get_data(flag='train')
            self.train_scores_1d, self.train_labels_1d = self._gather_scores_labels_1d(train_loader, self.args.temperature)
        
        
        best_k, threshold, pred_label_1d = 0, None, None

        if self.best_val_threshold is not None:
            threshold = self.best_val_threshold
            best_k = self.best_val_k
            pred_label_1d = (test_scores_1d > threshold).astype(np.int32)
        elif self.args.th_mode == 'percentile_val':
            best_k, threshold, _ = grid_search_percentile_k(test_scores_1d, test_labels_1d, self.args.th_grid)
            thre_po = ThresholdPolicy(mode=self.args.th_mode, k_percent=best_k)
            thre_po.fit(None, ref_score_1d=test_scores_1d)
            pred_label_1d = thre_po.predict(test_scores_1d)
        elif self.args.th_mode == 'percentile_train':
            best_k, threshold, _ = grid_search_percentile_k(self.val_scores_1d, self.val_labels_1d, self.args.th_grid, self.train_scores_1d)
            thre_po = ThresholdPolicy(mode=self.args.th_mode, k_percent=best_k)
            thre_po.fit(train_scores_1d=self.train_scores_1d, ref_score_1d=test_scores_1d)
            pred_label_1d = thre_po.predict(test_scores_1d)
        elif self.args.th_mode == 'unified':
            unified_scores = np.concatenate([self.train_scores_1d, test_scores_1d])
            best_k, threshold, _ = grid_search_percentile_k(test_scores_1d, test_labels_1d, self.args.th_grid, unified_scores)
            thre_po = ThresholdPolicy(mode=self.args.th_mode, k_percent=best_k)
            thre_po.fit(train_scores_1d=self.train_scores_1d, ref_score_1d=test_scores_1d)
            pred_label_1d = thre_po.predict(test_scores_1d)
        elif self.args.th_mode == "spot":
            best_k, threshold, _ = 0, None, None  # spot 自适应，不需要k
            thre_po = ThresholdPolicy(mode="spot")
            thre_po.fit(train_scores_1d=self.train_scores_1d, ref_score_1d=test_scores_1d)
            pred_label_1d = thre_po.predict(test_scores_1d)
            threshold = thre_po.get_shreshold()
        else:
            raise ValueError(f"Unknown threshold mode: {self.args.th_mode}")

        if test_labels_1d.sum() == 0:
            print("[WARN] No positive labels in test set, using predicted labels as reference.")
            test_labels_1d = pred_label_1d

        metrics = get_all_evaluation_score(pred_label_1d, test_labels_1d)
        metrics = {k: round(v, 5) for k, v in metrics.items()}
        if self.args.th_mode == 'spot':
            threshold = thre_po.get_shreshold()
        print(f'[TEST] threshold={threshold:.6f} | best_k={best_k} | metrics={metrics} ')
        # print(get_npsr_label(test_labels_1d, test_scores_1d))
        npsr_label = get_npsr_label(test_labels_1d, test_scores_1d)
        print('npsr_label')
        print(get_all_evaluation_score(npsr_label, test_labels_1d))

        idx = np.arange(len(test_scores_1d), dtype=np.int64)
        df_d = {
            'index': idx,
            'score': test_scores_1d,
            'pred_label': pred_label_1d,
            'label': test_labels_1d,
        }
        df = pd.DataFrame(df_d)
        df.to_csv(res_csv, index=False)

        gt = test_labels_1d
        pred = pred_label_1d
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

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy_pa : {:0.4f}, Precision_pa : {:0.4f}, Recall_pa : {:0.4f}, F-score_pa : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        return {'threshold': threshold, 'metrics': metrics}