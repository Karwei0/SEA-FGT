
# SEA-FGT: Frequency-Guided Transformer with Semantic Expert Augmentation

This repository provides the official PyTorch implementation of the paper: **"SEA-FGT: Semantic Expert Augmentation with Frequency-Guided Transformer for Multivariate Time-Series Anomaly Detection."**

---

## 🌟 Overview

**SEA-FGT** is a robust contrastive learning framework designed for multivariate time-series anomaly detection in real-world industrial systems. By leveraging a positive-sample-only strategy, the model effectively addresses three critical challenges:

* **Complex Inter-channel Dependencies:** Explored via the **Channel Correlation Exploration (CCE)** module.
* **Heterogeneous Temporal Semantics:** Captured through the **Semantic Expert Augmentation (SEA)** module.
* **Frequency-Domain Perturbations:** Modeled by the **Frequency-Guided Transformer (FGT)** to detect anomalies that are subtle in the time domain but prominent in the spectral domain.

### 📊 Advanced Evaluation Metrics

Beyond traditional point-wise metrics, this framework supports **range-oriented evaluation** to ensure fair and stable performance assessment:

* **Affiliation Metrics:** Affiliation Precision / Recall / F1
* **Volume Under Surface:** VUS-ROC / VUS-PR
* **Range-based AUC:** Range-AUC-ROC / Range-AUC-PR

---

## 📂 Repository Structure

```text
.
├── datasets/            # Data loading and pre-processing (PSM, SMD, etc.)
├── layers/              # Core components (FGT.py, SEA.py, CCE.py)
├── models/              # Full model architecture definitions
├── experiments/         # Experiment configurations and hyper-parameters
├── losses/              # Custom loss functions (Contrastive, Reconstruction)
├── metrics/             # Implementation of advanced evaluation metrics
├── scripts/             # Shell scripts for automated training and testing
├── utils/               # Utility functions (EarlyStopping, Data Scaling)
├── run.py               # Main entry point for the project
├── config.py            # Global configurations and path settings
└── requirements.txt     # Python dependencies

```

---

## 🛠️ Installation & Setup

### 1. Environment

We recommend using **Python 3.10** and **PyTorch 2.7.1**. You can install the dependencies via pip:

```bash
pip install -r requirements.txt

```

### 2. Data Preparation

Download the pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR). Place the folders (e.g., `MSL`, `SMAP`) into the `./datasets/` directory.

---

## 🚀 Getting Started

### Training & Evaluation

To train the model and evaluate it on a specific dataset:

```bash
python run.py --dataset MSL --exp_name quick_start --is_training 1

```

### Inference Only

To run inference using a pre-trained checkpoint (ensure `exp_name` matches the training session):

```bash
python run.py --dataset MSL --exp_name quick_start --is_training 0

```

### Batch Execution

Alternatively, use the provided shell scripts for one-click execution:

```bash
sh scripts/run.sh

```

---

## 🙏 Acknowledgements

We express our gratitude to the following projects for their contributions to the community:

* [DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector.git) for providing cleaned versions of the datasets.
* [TranAD](https://github.com/imperial-qore/TranAD.git), [RTdetector](https://github.com/CSUFUNLAB/RTdetector.git) and [TAB](https://github.com/decisionintelligence/TAB.git) for their excellent baseline implementations.
* Also pay homage to these contributors who publish their works online, used in this paper.
---

## 📜 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{sea_fgt2026,
  title={SEA-FGT: Semantic Expert Augmentation with Frequency-Guided Transformer for Multivariate Time-Series Anomaly Detection},
  author={},
  booktitle={},
  year={2026}
}

```