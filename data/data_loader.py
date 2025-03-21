import torch
from torch_geometric.datasets import QM9, MoleculeNet
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

def load_dataset(name, batch_size, mode="student"):
    """Loads the specified dataset and returns data loaders for training, validation, and testing."""
    if name == "qm9":
        dataset = QM9(root="./data")
        
        # Define indices for student and teacher targets based on the paper
        student_target_indices = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        teacher_target_indices = [0, 1, 2, 3, 4]  # First 5 targets for teacher model
        
        if mode == "student":
            dataset.data.y = dataset.data.y[:, student_target_indices]  # Select only these 10 targets
        elif mode == "teacher":
            dataset.data.y = dataset.data.y[:, teacher_target_indices]  # Select first 5 targets
        else:
            raise ValueError("Invalid mode. Choose from 'student' or 'teacher'.")
        
        num_targets = dataset.data.y.shape[1]
    elif name == "esol":
        dataset = MoleculeNet(root="./data", name="ESOL")
        num_targets = 1
    elif name == "freesolv":
        dataset = MoleculeNet(root="./data", name="FreeSolv")
        num_targets = 1
    else:
        raise ValueError("Invalid dataset name")
    
    # Get dataset length
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size  # Ensure sum matches total size

    # Define fixed indices to avoid breaking PyG behavior
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    # Use Subset to preserve dataset behavior
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)

    # Initialize DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, num_targets