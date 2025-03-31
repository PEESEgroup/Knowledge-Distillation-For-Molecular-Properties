# Knowledge Distillation for Molecular Property Prediction: A Scalability Analysis

This repository provides a framework for training teacher and student models for molecular property prediction using **Knowledge Distillation (KD)**. The implementation supports training on the **QM9, ESOL, and FreeSolv** datasets and utilizes **graph neural networks (GNNs)** like **SchNet, DimeNet++, and TensorNet**.

<p align="center">
  <img width="396" alt="image" src="[https://github.com/user-attachments/assets/c08ad6dc-22bf-4f2f-bfce-b1d43172b6ce](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties/blob/main/TOC.png)" />
</p>

## 📌 Features
- Train **teacher models** on QM9 (first 5 targets).
- Train **student models** with or without **knowledge distillation** on 10 different QM9 and two experimental datasets within MoleculeNet (ESOL/FreeSolv).
- Supports **SchNet, DimeNet++, and TensorNet** models.
- Implements **Uncertainity-weighted ensemble of L1 loss and Cosine Similarity loss** for our regression-based KD approach.
- **Hyperparameter tuning** using **Optuna**.
- **Early stopping & model checkpointing**.
- **Logging** of metrics and best models.

## 📂 Repository Structure
```
Knowledge_distillation_for_molecules/
│ 
│── architectures/               # Model architectures
│   ├── schnet.py                # SchNet architectures
│   ├── dimenetpp.py             # DimeNet++ architectures
│   ├── tensornet.py             # TensorNet architectures
│── training/                    # Training scripts
│   ├── train_teacher.py         # Train teacher models
│   ├── train_student.py         # Train student models (with/without KD)
│   ├── total_loss.py            # Loss function (L1 + Cosine similarity)
│   ├── optimize_hyperparams.py  # Optuna-based hyperparameter tuning  ✅
├── data/                        # Data processing
│   ├── data_loader.py           # Loads QM9, ESOL, FreeSolv datasets
├── utils/                       # Helper functions
│   ├── train_utils.py           # Metrics, reproducibility
│   ├── config.py                # Centralized hyperparameters
│── results/                          # Logs, saved models (will be created during model checkpoint saving)
│── README.md                         # Documentation
│── requirements.txt                  # Dependencies
```

## 🚀 Installation

### **Using mamba for dependency management**
To set up the environment and install dependencies, including `torchmd-net`:

```bash
mamba create -n molecular-learning python=3.9
mamba activate molecular-learning
mamba install pytorch torchvision torchaudio -c pytorch
mamba install pyg -c pyg
mamba install torchmd-net -c conda-forge
pip install -r requirements.txt
```

## 🔥 Training
### **1️⃣ Train the Teacher Model**
Run the following command to train a **teacher model**:
```bash
python training/train_teacher.py
```
This trains a **SchNet/DimeNet++/TensorNet** model on **QM9** (first 5 targets).

### **2️⃣ Train the Student Model (With or Without KD)**
To train the **student model**, run:
```bash
python training/train_student.py
```
- **With KD:** The script will use the pre-trained **teacher model**.
- **Without KD:** Set `USE_KD = False` in `config.py`.

## 🔍 Hyperparameter Tuning
To tune **learning rate, batch size, and alpha** using **Optuna**, run:
```bash
python training/optimize_hyperparams.py
```
- The best hyperparameters will be **saved in `config_optimized.json`**.
- To apply them automatically, ensure `PERFORM_TUNING = False` in `config.py`.

## 📊 Logging & Results
- **Best models** are saved in `results/`.
- **Metrics (train loss, val loss, R²)** are logged in `config.LOG_PATH`.

## 📥 Pretrained Models
- Pretrained teacher and student models for QM9, ESOL, and FreeSolv can be downloaded here - [pretrained models](https://drive.google.com/drive/folders/1k_N6Cswk57DlxprMFuArh-oaxTz_V-xi?usp=sharing)

## 📜 Citation
- If you find this repository useful, please cite our work:
  
```
@article{molKD2025,
  author    = {Sheshanarayana, Rahul and You, Fengqi},
  title     = {Knowledge Distillation for Molecular Property Prediction: A Scalability Analysis},
  journal   = {Advanced Science},
  volume    = {n/a},
  number    = {n/a},
  pages     = {# insert after published #},
  keywords  = {materials informatics, graph neural networks, scalability, knowledge distillation},
  doi       = {# insert after published #},
  url       = {# insert after published #},
  eprint    = {# insert after published #},
  abstract  = {# insert after published #}
}
```
---

Feel free to contribute or raise issues! 🚀
