# SEA-FGT：语义专家增强与频率引导 Transformer 的多变量时间序列异常检测

[English](README.md) | [中文说明](README_CN.md)

---

SEA-FGT 是一种**完全基于正样本的对比学习框架**，用于多变量时间序列异常检测。  
该方法面向真实工业系统，旨在联合建模以下关键挑战：

- 通道间复杂且动态的相关依赖  
- 不同通道与序列之间的异构时间语义  
- 异常事件在频域中引起的结构性扰动  

本仓库提供 SEA-FGT 的**官方 PyTorch 实现**，包含模型训练、评估流程以及论文中使用的可视化脚本。

---

## ✨ 方法特点

- **通道相关性探索（Channel Correlation Exploration, CCE）**  
  基于频域相干性分析，显式建模多变量时间序列中的通道间依赖关系。

- **语义专家增强（Semantic Expert Augmentation, SEA）**  
  采用类似 Mixture-of-Experts 的结构，引入多种具有不同时间建模归纳偏置的语义专家，实现结构化增强。

- **频率引导 Transformer（Frequency-Guided Transformer, FGT）**  
  利用谱熵信息，自适应调节注意力机制，使模型更关注异常敏感的通道与时间片段。

- **纯正样本对比学习**  
  训练过程中不依赖显式负样本构造，避免负样本采样偏置，同时简化训练流程。

---

## 📁 项目结构

```
.
├── datasets/                 # 数据集加载与预处理
│   ├── PSM/
│   ├── SMD/
│   ├── SMAP/
│   └── SWaT/
├── layers/                   # 核心模型组件
│   ├── FGT.py                # 频率引导 Transformer
│   ├── SEA.py                # 语义专家增强模块
│   ├── CCE.py                # 通道相关性探索模块
│   └── ...
├── models/                   # 完整模型定义
├── scripts/                  # 训练与评估脚本
├── outputs/                  # 实验日志与可视化结果
├── checkpoints/              # 模型权重
├── requirements.txt
└── README.md
```

---

## ⚙️ 环境配置

推荐使用 Conda 创建独立环境：

```bash
conda create -n sea_fgt python=3.9
conda activate sea_fgt
pip install -r requirements.txt
```

---

## 🚀 模型训练

示例：在 **SMD** 数据集上训练 SEA-FGT

```bash
python scripts/train.py \
  --dataset SMD \
  --use_cce \
  --use_sea \
  --use_fgt \
  --top_k 2
```

### 主要参数说明

- `--dataset`：数据集名称（`PSM`, `SMD`, `SMAP`, `SWaT`）
- `--use_cce`：是否启用通道相关性探索模块
- `--use_sea`：是否启用语义专家增强模块
- `--use_fgt`：是否启用频率引导 Transformer
- `--top_k`：SEA 中选择的专家数量
- `--lambda_load`：专家负载均衡损失的权重系数

---

## 📊 模型评估

```bash
python scripts/eval.py \
  --dataset SMD \
  --checkpoint checkpoints/SMD_best.pt
```

评估流程支持多种**面向异常区间的高级指标**，包括：

- Affiliation Precision / Recall / F1  
- Range-AUC-ROC / Range-AUC-PR  
- VUS-ROC / VUS-PR  

这些指标能够更加公平、稳定地评估连续异常检测性能。

---

## 📈 可视化分析

仓库中提供了多种可视化脚本，用于复现实验分析结果，包括：

- 谱熵变化与异常证据分析  
- 通道相干矩阵的 k-sparse 敏感性分析  
- 语义专家结构与消融实验  
- 专家负载均衡动态分析  
- Precision–Recall 等值线可视化  

所有结果默认保存在 `outputs/` 目录下。

---

## 🧠 关于训练稳定性

SEA-FGT 采用**纯正样本对比学习**，理论上可能存在表示坍缩风险。  
在实际训练中，我们通过以下机制有效缓解该问题：

- 基于验证集性能的 Early Stopping  
- 由异构语义专家引入的结构多样性  
- 专家负载均衡正则化约束  

在所有实验数据集上均未观察到表示坍缩现象。

---

## 🔁 可复现性说明

- 所有实验均使用固定随机种子  
- 关键超参数已在论文中给出  
- 仓库提供完整脚本以复现实验结果与消融分析  

---

## 📜 引用

如果本工作对你的研究有所帮助，欢迎引用：

```bibtex
@inproceedings{sea_fgt,
  title={SEA-FGT: Semantic Expert Augmentation with Frequency-Guided Transformer for Multivariate Time-Series Anomaly Detection},
  author={},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

---

## 📬 联系方式

如有问题或希望进一步交流，欢迎在仓库中提交 Issue。
