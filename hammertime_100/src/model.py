"""
Module M6: Model Definition
Responsibility: Define the dynamic Spatio-Temporal Graph Attention Network (ST-GNN).
Stages: Linear projection -> GATConv x2 -> causal temporal module -> MLP head.
Standard: ASD-STE100 Plain English
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.model")


class CausalGatedResidualBlock(nn.Module):
    """Gated dilated temporal convolution with left-only padding and a residual."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive.")
        self.dilation = int(dilation)
        self.kernel_size = 3
        self.causal_conv = nn.Conv1d(channels, channels * 2, self.kernel_size,
                                     dilation=self.dilation, padding=0)
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        left_padding = (self.kernel_size - 1) * self.dilation
        filtered = self.causal_conv(F.pad(sequence, (left_padding, 0)))
        content, gate = filtered.chunk(2, dim=1)
        gated = torch.tanh(content) * torch.sigmoid(gate)
        return sequence + self.dropout(self.projection(gated))


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HammerTimeSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for F1 Traffic Delay Prediction.
    Stage 1: Input projection (Linear)
    Stage 2: Spatial graph attention (2x GATConv with dynamic RBF edge weights)
    Stage 3: Legacy Conv1d or an optional gated dilated causal residual block stack
    Stage 4: MLP head predicting delta t_sector per car
    """
    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        edge_dim: int = 1,
        conv1d_kernel: int = 3,
        dropout: float = 0.1,
        temporal_mode: str = "conv",
        temporal_dilations: list[int] | tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        if hidden_dim <= 0 or gat_heads <= 0 or hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by gat_heads.")
        if conv1d_kernel <= 0:
            raise ValueError("conv1d_kernel must be positive.")
        if temporal_mode not in ("conv", "gated_dilated"):
            raise ValueError("temporal_mode must be 'conv' or 'gated_dilated'.")
        if (not isinstance(temporal_dilations, (list, tuple)) or not temporal_dilations
                or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                       for item in temporal_dilations)):
            raise ValueError("temporal_dilations must contain positive integers.")
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

        # Stage 3: Legacy convolution stays parameter-identical for old checkpoints.
        self.temporal_mode = temporal_mode
        if temporal_mode == "conv":
            self.tconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv1d_kernel, padding=0)
            self.temporal_blocks = nn.ModuleList()
        else:
            self.tconv = None
            self.temporal_blocks = nn.ModuleList(
                CausalGatedResidualBlock(hidden_dim, dilation, dropout)
                for dilation in temporal_dilations
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
        edge_index = edge_index.to(device)
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float, device=device)
        else:
            edge_attr = edge_attr.to(device)

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
            device = self.proj.weight.device
            x = g.x.to(device=device, dtype=self.proj.weight.dtype)
            N = x.size(0)
            
            # 1. Project node features
            h = torch.relu(self.proj(x))

            # 2. Add self-loops for robust message passing
            edge_idx, edge_attr = self._ensure_self_loops(
                N, getattr(g, "edge_index", None), getattr(g, "edge_attr", None), device
            )
            edge_idx = edge_idx.to(device)
            edge_attr = edge_attr.to(device=device, dtype=h.dtype)

            # 3. Two GAT layers
            h = torch.relu(self.gat1(h, edge_idx, edge_attr))
            h = torch.relu(self.gat2(h, edge_idx, edge_attr))
            outs.append(h)  # [N, hidden_dim]

        # Stack over time: [N, hidden_dim, W]
        z = torch.stack(outs, dim=2)

        if self.temporal_mode == "conv":
            # Preserve the legacy causal temporal path exactly.
            z = F.pad(z, (self.tconv.kernel_size[0] - 1, 0))
            z = torch.relu(self.tconv(z))
        else:
            for temporal_block in self.temporal_blocks:
                z = temporal_block(z)

        # Take last time step
        z_last = z[:, :, -1]  # [N, hidden_dim]

        # MLP head -> [N]
        out = self.head(z_last).squeeze(-1)
        return out

    def _forward_batched_windows(self, batch_windows):
        """Run spatial message passing over one disjoint union of all B*W graphs."""
        batch_size = len(batch_windows)
        window_size = len(batch_windows[0])
        if window_size == 0 or any(len(window) != window_size for window in batch_windows):
            raise ValueError("Batched windows must have the same non-zero temporal length.")
        first_x = batch_windows[0][0].x
        num_nodes = int(first_x.size(0))
        if num_nodes <= 0 or any(int(snapshot.x.size(0)) != num_nodes
                                 for window in batch_windows for snapshot in window):
            raise ValueError("Batched windows require the same positive node count in every snapshot.")

        device = self.proj.weight.device
        dtype = self.proj.weight.dtype
        graphs = []
        for window in batch_windows:
            for snapshot in window:
                x = snapshot.x.to(device=device, dtype=dtype)
                edge_index, edge_attr = self._ensure_self_loops(
                    num_nodes, getattr(snapshot, "edge_index", None),
                    getattr(snapshot, "edge_attr", None), device
                )
                graphs.append(Data(x=x, edge_index=edge_index,
                                   edge_attr=edge_attr.to(device=device, dtype=dtype)))

        disjoint = Batch.from_data_list(graphs).to(device)
        h = torch.relu(self.proj(disjoint.x))
        h = torch.relu(self.gat1(h, disjoint.edge_index, disjoint.edge_attr))
        h = torch.relu(self.gat2(h, disjoint.edge_index, disjoint.edge_attr))
        h = h.reshape(batch_size, window_size, num_nodes, self.hidden_dim)
        z = h.permute(0, 2, 3, 1).contiguous()
        z = z.reshape(batch_size * num_nodes, self.hidden_dim, window_size)

        if self.temporal_mode == "conv":
            z = F.pad(z, (self.tconv.kernel_size[0] - 1, 0))
            z = torch.relu(self.tconv(z))
        else:
            for temporal_block in self.temporal_blocks:
                z = temporal_block(z)
        z_last = z[:, :, -1].reshape(batch_size, num_nodes, self.hidden_dim)
        return self.head(z_last).squeeze(-1)

    def forward(self, batch_windows):
        """
        Accepts either:
          - A single list of W Data objects -> returns [N]
          - A list of B items (each is a list of W Data objects) -> returns [B, N]
        """
        if not batch_windows:
            raise ValueError("Model input must contain at least one graph window.")
        if isinstance(batch_windows[0], (list, tuple)):
            return self._forward_batched_windows(batch_windows)
        return self.forward_window(batch_windows)


def create_model(config: dict) -> HammerTimeSTGNN:
    """Build model instance from configuration dict."""
    seed = int(config.get("data", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m_cfg = config["model"]
    return HammerTimeSTGNN(
        in_dim=m_cfg.get("in_dim", 9),
        hidden_dim=m_cfg.get("hidden_dim", 64),
        gat_heads=m_cfg.get("gat_heads", 4),
        edge_dim=m_cfg.get("edge_dim", 1),
        conv1d_kernel=m_cfg.get("conv1d_kernel", 3),
        dropout=m_cfg.get("dropout", 0.1),
        temporal_mode=m_cfg.get("temporal_mode", "conv"),
        temporal_dilations=m_cfg.get("temporal_dilations", [1, 2, 4]),
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
