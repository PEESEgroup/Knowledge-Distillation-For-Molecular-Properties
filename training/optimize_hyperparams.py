import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from utils import config
from torch_geometric.loader import DataLoader
from data.data_loader import load_dataset
from architectures import SchNet, DimeNetPPModel, TensorNet
from training.total_loss import TotalLoss
from utils.train_utils import compute_metrics

# Ensure save directory exists
os.makedirs(config.SAVE_DIR, exist_ok=True)

# Load dataset
train_loader, val_loader, test_loader, num_targets = load_dataset(config.DATASET_NAME, config.BATCH_SIZE, mode="student")
num_teacher_targets = 5  # Hardcoded for now

def get_model(model_name, num_targets):
    if model_name == "schnet":
        return SchNet(num_targets=num_targets).to(config.DEVICE)
    elif model_name == "dimenetpp":
        return DimeNetPPModel(out_channels=num_targets).to(config.DEVICE)
    elif model_name == "tensornet":
        return TensorNet(num_targets=num_targets).to(config.DEVICE)
    else:
        raise ValueError("Invalid model name")

def train_one_epoch(student, teacher, train_loader, optimizer, criterion):
    """Training loop for one epoch."""
    student.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(config.DEVICE)
        optimizer.zero_grad()
        
        student_output, student_emb = student(data.z, data.pos, data.batch)
        
        if config.USE_KD:
            with torch.no_grad():
                _, teacher_emb = teacher(data.z, data.pos, data.batch)
            loss = criterion(student_output, student_emb, teacher_emb, data.y)
        else:
            loss = criterion(student_output, student_emb, student_emb, data.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(student, val_loader, criterion):
    """Validation loop."""
    student.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(config.DEVICE)
            output, _ = student(data.z, data.pos, data.batch)
            loss = criterion(output, output, output, data.y)
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    return total_loss / len(val_loader), metrics["R2"]

# Define objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Load dataset with suggested batch size
    train_loader, val_loader, _, num_targets = load_dataset(config.DATASET_NAME, batch_size, mode="student")
    
    # Load models
    teacher = SchNet(num_targets=5).to(config.DEVICE)
    student = SchNet(num_targets=num_targets).to(config.DEVICE)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = TotalLoss()

    best_val_r2 = -float("inf")
    
    # Train for a few epochs (e.g., 10 for tuning)
    for epoch in range(1):
        train_one_epoch(student, teacher, train_loader, optimizer, criterion)
        _, val_r2 = validate(student, val_loader, criterion)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
    
    return best_val_r2

# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)
