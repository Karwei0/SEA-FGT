# SEA-FGT: Semantic Expert Augmentation with Frequency-Guided Transformer

[English](README.md) | [中文说明](README_CN.md)

---

SEA-FGT is a **fully positive-sample-based contrastive learning framework** for multivariate time-series anomaly detection.  
It is designed to address key challenges in real-world industrial systems, including **complex inter-channel dependencies**, **heterogeneous temporal semantics**, and **anomaly-induced frequency-domain perturbations**.

This repository provides the **official PyTorch implementation** of SEA-FGT, including training, evaluation, and visualization code used in our paper.

---

## 🔍 Key Features

- **Channel Correlation Exploration (CCE)**  
  Explicitly captures inter-channel dependencies using frequency-domain coherence analysis.

- **Semantic Expert Augmentation (SEA)**  
  A Mixture-of-Experts–style augmentation module composed of heterogeneous semantic experts with different temporal inductive biases.

- **Frequency-Guided Transformer (FGT)**  
  Integrates spectral entropy to adaptively modulate attention towards anomaly-sensitive channels.

- **Positive-only Contrastive Learning**  
  Eliminates the need for explicit negative sampling, reducing sampling bias and simplifying training.

---

## 📁 Project Structure

```
.
├── datasets/                 # Dataset loaders and preprocessing
│   ├── PSM/
│   ├── SMD/
│   ├── SMAP/
│   └── SWaT/
├── layers/                   # Core model components
│   ├── FGT.py                # Frequency-Guided Transformer
│   ├── SEA.py                # Semantic Expert Augmentation
│   ├── CCE.py                # Channel Correlation Exploration
│   └── ...
├── models/                   # Full model definitions
├── scripts/                  # Training and evaluation scripts
├── outputs/                  # Saved logs and figures
├── checkpoints/              # Model checkpoints
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

```bash
conda create -n sea_fgt python=3.9
conda activate sea_fgt
pip install -r requirements.txt
```

---

## 🚀 Training

Example: training SEA-FGT on the **SMD** dataset

```bash
python scripts/train.py \
  --dataset SMD \
  --use_cce \
  --use_sea \
  --use_fgt \
  --top_k 2
```

---

## 📊 Evaluation

```bash
python scripts/eval.py \
  --dataset SMD \
  --checkpoint checkpoints/SMD_best.pt
```

---

## 📜 Citation

```bibtex
@inproceedings{sea_fgt,
  title={SEA-FGT: Semantic Expert Augmentation with Frequency-Guided Transformer for Multivariate Time-Series Anomaly Detection},
  author={},
  booktitle={},
  year={2026}
}
```
