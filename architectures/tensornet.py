# Implementation of TensorNet inspired by TorchMD-Net

import torch
from typing import Optional, Tuple
from torch import Tensor, nn
from torchmdnet.models.utils import (
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
)

__all__ = ["TensorNet"]

def vector_to_skewtensor(vector):
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero, -vector[:, 2], vector[:, 1],
            vector[:, 2], zero, -vector[:, 0],
            -vector[:, 1], vector[:, 0], zero,
        ), dim=1,
    ).view(-1, 3, 3)
    return tensor.squeeze(0)

def vector_to_symtensor(vector):
    """Creates a symmetric traceless tensor from the outer product of a vector with itself."""
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[..., None, None] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S

def decompose_tensor(tensor):
    """Full tensor decomposition into irreducible components."""
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[..., None, None] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S

def tensor_norm(tensor):
    """Computes Frobenius norm."""
    return (tensor**2).sum((-2, -1))

class TensorEmbedding(nn.Module):
    """Tensor embedding layer."""
    def __init__(
        self, hidden_channels, num_rbf, activation, max_z=128, dtype=torch.float32
    ):
        super(TensorEmbedding, self).__init__()
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(max_z, hidden_channels, dtype=dtype)
        self.act = activation()
        self.reset_parameters()
    
    def reset_parameters(self):
        self.emb.reset_parameters()
    
    def forward(self, z: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_vec: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.emb(z)
        return x

class Interaction(nn.Module):
    """Interaction layer for TensorNet."""
    def __init__(self, num_rbf, hidden_channels, activation):
        super(Interaction, self).__init__()
        self.hidden_channels = hidden_channels
        self.act = activation()
        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear.reset_parameters()
    
    def forward(self, X: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        return self.act(self.linear(X))

class TensorNet(nn.Module):
    """TensorNet architecture inspired by TensorNet: Cartesian Tensor Representations for Molecular Potentials.
    Implements interaction layers, embeddings, and tensor operations for molecular systems.
    """
    def __init__(
        self,
        hidden_channels=128,
        num_layers=2,
        num_rbf=32,
        rbf_type="expnorm",
        activation="silu",
        cutoff_lower=0,
        cutoff_upper=4.5,
        max_num_neighbors=64,
        max_z=128,
        static_shapes=True,
        check_errors=True,
        dtype=torch.float32,
        num_targets=1,
    ):
        super(TensorNet, self).__init__()
        
        # Activation function mapping
        assert rbf_type in rbf_class_mapping, f'Unknown RBF type "{rbf_type}"'
        assert activation in act_class_mapping, f'Unknown activation function "{activation}"'
        
        act_class = act_class_mapping[activation]
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.static_shapes = static_shapes
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        # Distance expansion layer
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff_lower, cutoff_upper, num_rbf)
        
        # Tensor embedding layer
        self.tensor_embedding = TensorEmbedding(hidden_channels, num_rbf, act_class, max_z, dtype)
        
        # Interaction layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Interaction(num_rbf, hidden_channels, act_class))
        
        # Output layers
        self.linear = nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype)
        self.out_norm = nn.LayerNorm(3 * hidden_channels, dtype=dtype)
        self.act = act_class()
        self.final_linear = nn.Linear(hidden_channels, num_targets)
        
        # Distance module
        self.distance = OptimizedDistance(
            cutoff_lower, cutoff_upper, max_num_pairs=-max_num_neighbors, return_vecs=True,
            loop=True, check_errors=check_errors, resize_to_fit=not static_shapes,
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.linear.reset_parameters()
        self.out_norm.reset_parameters()
        self.final_linear.reset_parameters()
    
    def forward(self, z: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass: Computes the output properties and embeddings."""
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        
        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)
        for layer in self.layers:
            X = layer(X, edge_index, edge_weight, edge_attr)
        
        I, A, S = decompose_tensor(X)
        x = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
        x = self.out_norm(x)
        x = self.act(self.linear(x))
        emb = x
        x = self.final_linear(x)
        return x, emb
