import torch
import json
import os

# General Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Default Training Hyperparameters (will be updated by Optuna tuning if enabled)
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 1
EARLY_STOPPING_PATIENCE = 5

# Knowledge Distillation Parameters
USE_KD = True  # Toggle KD on/off

# Dataset and Model Selection
DATASET_NAME = "qm9"  # Choose from "qm9", "esol", "freesolv"
MODEL_NAME = "schnet"  # Choose from "schnet", "dimenetpp", "tensornet"

# Paths for Saving Models and Logs
SAVE_DIR = "results/"
BEST_TEACHER_MODEL_PATH = f"{SAVE_DIR}/{MODEL_NAME}_{DATASET_NAME}_teacher.pth"
BEST_STUDENT_MODEL_PATH = f"{SAVE_DIR}/{MODEL_NAME}_{DATASET_NAME}_student.pth"
LOG_PATH = f"{SAVE_DIR}/{MODEL_NAME}_{DATASET_NAME}_log.json"

# Option to enable hyperparameter tuning
PERFORM_TUNING = False  # Set to True if tuning is required

# Load optimized hyperparameters if available and tuning is not requested
CONFIG_OPTIMIZED_PATH = "config_optimized.json"
if os.path.exists(CONFIG_OPTIMIZED_PATH) and not PERFORM_TUNING:
    with open(CONFIG_OPTIMIZED_PATH, "r") as f:
        optimized_params = json.load(f)
        LEARNING_RATE = optimized_params.get("lr", LEARNING_RATE)
        BATCH_SIZE = optimized_params.get("batch_size", BATCH_SIZE)
        ALPHA = optimized_params.get("alpha", ALPHA)
