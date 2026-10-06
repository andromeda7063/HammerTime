"""
Module M6: Model Definition
Responsibility: Define the dynamic Spatio-Temporal Graph Attention Network (ST-GNN).
Stages: Linear projection -> GATConv x2 (edge_dim=1, heads=4) -> Conv1d (time) -> MLP head.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.model")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HammerTimeSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for F1 Traffic Delay Prediction.
    Stage 1: Input projection (Linear)
    Stage 2: Spatial graph attention (2x GATConv with dynamic RBF edge weights)
    Stage 3: Temporal convolution across W snapshots (Conv1d)
    Stage 4: MLP head predicting delta t_sector per car
    """
    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        edge_dim: int = 1,
        conv1d_kernel: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        head_dim = hidden_dim // gat_heads

        # Stage 1: Linear Input Projection
        self.proj = nn.Linear(in_dim, hidden_dim)

        # Stage 2: Spatial Message Passing (GAT with edge features)
        # Note: add_self_loops=True with fill_value=1.0 ensures zero-edge snapshots flow smoothly
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=gat_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=gat_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False
        )

        # Stage 3: Temporal Convolution over W-window
        self.tconv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv1d_kernel,
            padding=conv1d_kernel // 2
        )

        # Stage 4: MLP Regression Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _ensure_self_loops(self, num_nodes: int, edge_index: torch.Tensor, edge_attr: torch.Tensor, device):
        """
        Adds self-loops to guaranteed graph connectivity even when active edges are 0.
        Self-loop weight is 1.0 (zero gap to self).
        """
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
        loop_attr = torch.ones((num_nodes, 1), dtype=torch.float, device=device)

        if edge_index is None or edge_index.size(1) == 0:
            return loop_index, loop_attr

        combined_index = torch.cat([edge_index, loop_index], dim=1)
        combined_attr = torch.cat([edge_attr, loop_attr], dim=0)
        return combined_index, combined_attr

    def forward_window(self, snapshots: list):
        """
        Forward pass for a single sequence of W snapshots.
        snapshots: list of W Data objects, each with x: [N, in_dim]
        Returns: [N] predicted delay per car
        """
        outs = []
        for g in snapshots:
            device = g.x.device
            N = g.x.size(0)
            
            # 1. Project node features
            h = torch.relu(self.proj(g.x))

            # 2. Add self-loops for robust message passing
            edge_idx, edge_attr = self._ensure_self_loops(N, g.edge_index, g.edge_attr, device)

            # 3. Two GAT layers
            h = torch.relu(self.gat1(h, edge_idx, edge_attr))
            h = torch.relu(self.gat2(h, edge_idx, edge_attr))
            outs.append(h)  # [N, hidden_dim]

        # Stack over time: [N, hidden_dim, W]
        z = torch.stack(outs, dim=2)

        # Temporal convolution over W window
        z = torch.relu(self.tconv(z))

        # Take last time step
        z_last = z[:, :, -1]  # [N, hidden_dim]

        # MLP head -> [N]
        out = self.head(z_last).squeeze(-1)
        return out

    def forward(self, batch_windows):
        """
        Accepts either:
          - A single list of W Data objects -> returns [N]
          - A list of B items (each is a list of W Data objects) -> returns [B, N]
        """
        if isinstance(batch_windows[0], list):
            # Batched execution
            preds = [self.forward_window(win) for win in batch_windows]
            return torch.stack(preds, dim=0)
        else:
            return self.forward_window(batch_windows)


def create_model(config: dict) -> HammerTimeSTGNN:
    """Build model instance from configuration dict."""
    m_cfg = config["model"]
    return HammerTimeSTGNN(
        in_dim=m_cfg.get("in_dim", 9),
        hidden_dim=m_cfg.get("hidden_dim", 64),
        gat_heads=m_cfg.get("gat_heads", 4),
        edge_dim=m_cfg.get("edge_dim", 1),
        conv1d_kernel=m_cfg.get("conv1d_kernel", 3),
        dropout=m_cfg.get("dropout", 0.1)
    )


if __name__ == "__main__":
    from torch_geometric.data import Data
    cfg = load_config()
    model = create_model(cfg)
    logger.info(f"Initialized HammerTime ST-GNN with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Test forward with dummy zero-edge window (Immutable Law 9 test)
    dummy_snaps = [
        Data(
            x=torch.randn(20, 9),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            mask=torch.ones(20, dtype=torch.bool)
        )
        for _ in range(10)
    ]
    out = model(dummy_snaps)
    logger.info(f"Zero-edge test forward shape: {out.shape} (Expected: [20])")
    assert out.shape == torch.Size([20]), "Model output shape mismatch!"
    logger.info("Immutable Law 9 (Degraded / zero-edge forward flow) test passed.")
