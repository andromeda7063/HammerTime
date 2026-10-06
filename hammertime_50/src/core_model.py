"""
HammerTime 50%: Core ST-GNN Model
Lightweight spatio-temporal graph neural network for fast baseline prediction.
Standard: ASD-STE100 Plain English
"""

import torch
import torch.nn as nn


class CoreSTGNN(nn.Module):
    """
    Lightweight Graph Neural Network for core demonstration.
    """
    def __init__(self, in_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

        # Basic linear message passing with edge weights
        self.edge_proj = nn.Linear(1, hidden_dim)
        self.msg_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # MLP Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward_snapshot(self, x, edge_index, edge_attr):
        """Pass messages for one snapshot."""
        N = x.size(0)
        h = torch.relu(self.proj(x))  # [N, hidden]

        # Aggregate neighbor disturbance
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            e_emb = torch.relu(self.edge_proj(edge_attr))
            msg = torch.relu(self.msg_layer(torch.cat([h[src], e_emb], dim=-1)))

            # Scatter add to destination nodes
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            h = h + agg

        return h

    def forward(self, window_snaps: list):
        """
        window_snaps: list of W CoreGraphSnapshot objects.
        Returns: [N] predicted delta t_sector
        """
        snap_outs = []
        for s in window_snaps:
            h = self.forward_snapshot(s.x, s.edge_index, s.edge_attr)
            snap_outs.append(h)

        # [N, hidden, W]
        stacked = torch.stack(snap_outs, dim=2)
        pooled = self.temporal_pool(stacked).squeeze(-1)  # [N, hidden]
        out = self.head(pooled).squeeze(-1)              # [N]
        return out
