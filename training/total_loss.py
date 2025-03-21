import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted loss combining L1 loss for regression and cosine similarity loss for embedding alignment."""

    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()
        self.cosine_criterion = nn.CosineSimilarity(dim=1)

        # Learnable log-variance parameters for uncertainty weighting
        self.log_sigma1 = nn.Parameter(torch.zeros(1))  # Regression loss uncertainty
        self.log_sigma2 = nn.Parameter(torch.zeros(1))  # KD loss uncertainty

    def forward(self, student_output, student_emb, teacher_emb, true_labels):
        task_loss = self.l1_criterion(student_output, true_labels)  # Supervised regression loss
        kd_loss = 1 - self.cosine_criterion(student_emb, teacher_emb).mean()  # Embedding similarity loss

        # Convert log-variance to standard deviation (ensures positivity)
        sigma1 = torch.exp(self.log_sigma1)
        sigma2 = torch.exp(self.log_sigma2)

        # Uncertainty-weighted total loss
        total_loss = (task_loss / (2 * sigma1**2)) + (kd_loss / (2 * sigma2**2)) + self.log_sigma1 + self.log_sigma2
        return total_loss

