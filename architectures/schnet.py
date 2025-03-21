import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor


class SchNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        num_targets: int = 5,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = aggr_resolver(readout)
        self.sum_aggr = SumAggregation()
        self.num_targets = num_targets

        # Embedding layers
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction layers
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        # Final layers
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, num_targets)

        # Parameter initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, z: Tensor, pos: Tensor, batch: OptTensor = None) -> Tensor:
        """
        Forward pass.

        Args:
            z (Tensor): Atomic number of each atom.
            pos (Tensor): Atomic positions.
            batch (Tensor, optional): Batch indices.

        Returns:
            out (Tensor): Model predictions.
            emb_aggr (Tensor): Aggregated embeddings for KD.
        """
        batch = torch.zeros_like(z) if batch is None else batch

        # Compute initial embeddings
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=32)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        # Apply interaction layers
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Extract embeddings for KD
        emb_aggr = self.sum_aggr(h, batch, dim=0)

        # Final transformation
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # Readout
        out = self.readout(h, batch, dim=0)

        return out, emb_aggr  # Returning both output & embeddings

    def __repr__(self) -> str:
        return f'SchNet(hidden_channels={self.hidden_channels}, num_interactions={self.num_interactions}, num_gaussians={self.num_gaussians}, cutoff={self.cutoff})'


class InteractionBlock(torch.nn.Module):
    """SchNet Interaction Block"""
    def __init__(self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    """Continuous-Filter Convolution"""
    def __init__(self, in_channels: int, out_channels: int, num_filters: int, nn: Sequential, cutoff: float):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    """Expands scalar distances into Gaussian basis functions."""
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    """Softplus activation with a shift for stability."""
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift
