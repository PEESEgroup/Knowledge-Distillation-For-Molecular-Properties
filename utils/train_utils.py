import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics: R², MAE, MSE."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred, multioutput='uniform_average')
    }
