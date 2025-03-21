import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
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
# Load student model
def get_model(model_name, num_targets):
    if model_name == "schnet":
        return SchNet(num_targets=num_targets).to(config.DEVICE)
    elif model_name == "dimenetpp":
        return DimeNetPPModel(out_channels=num_targets).to(config.DEVICE)
    elif model_name == "tensornet":
        return TensorNet(num_targets=num_targets).to(config.DEVICE)
    else:
        raise ValueError("Invalid model name")

student = get_model(config.MODEL_NAME, num_targets)

# Load teacher model (for KD)
if config.USE_KD:
    teacher = get_model(config.MODEL_NAME, num_teacher_targets)
    teacher.load_state_dict(torch.load(f"results/{config.MODEL_NAME}_{config.DATASET_NAME}_teacher", weights_only=True, map_location=torch.device('cpu')))
    teacher.to(config.DEVICE)
    teacher.eval()

# Loss and optimizer
criterion = TotalLoss()
optimizer = optim.Adam(student.parameters(), lr=config.LEARNING_RATE)

# Tracking best model
best_val_loss = float("inf")
best_epoch = 0
log_data = []

def train_one_epoch():
    """Training loop for one epoch."""
    student.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(config.DEVICE)
        optimizer.zero_grad()
        
        # Forward pass
        student_output, student_emb = student(data.z, data.pos, data.batch)
        
        if config.USE_KD:
            with torch.no_grad():
                _, teacher_emb = teacher(data.z, data.pos, data.batch)
            loss = criterion(student_output, student_emb, teacher_emb, data.y)  # KD loss
        else:
            loss = criterion(student_output, student_emb, student_emb, data.y)  # No KD, just supervised loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
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
    # Compute R² score
    metrics = compute_metrics(all_targets, all_preds)
    return total_loss / len(val_loader), metrics["R2"]

def test():
    """Test loop to evaluate final model performance."""
    student.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(config.DEVICE)
            output, _ = student(data.z, data.pos, data.batch)
            loss = criterion(output, output, output, data.y)
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    # Compute R² score
    metrics = compute_metrics(all_targets, all_preds)
    return total_loss / len(test_loader), metrics["R2"]

# Training loop
for epoch in range(config.EPOCHS):
    train_loss = train_one_epoch()
    val_loss, val_r2 = validate()
    
    print(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(student.state_dict(), config.BEST_STUDENT_MODEL_PATH)
        print(f"Saved Best Model at {config.BEST_STUDENT_MODEL_PATH}")
    
    # Log results
    log_data.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_r2": val_r2})
    with open(config.LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=4)
    
    # Early stopping
    if epoch - best_epoch >= config.EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

# Final Test Evaluation
test_loss, test_r2 = test()
print(f"Final Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")
