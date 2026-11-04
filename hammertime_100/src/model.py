"""
Module M6: Model Definition
Responsibility: Define the dynamic Spatio-Temporal Graph Network (ST-GNN).
Stages: Linear projection -> configured GAT stack or typed physical-edge pass -> causal temporal module -> sector head,
with an optional additive own-car MLP bypass from the final input frame.
Standard: ASD-STE100 Plain English
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GATv2Conv
import yaml
from physical_graph import (EDGE_FEATURES_BY_SCHEMA, TYPED_EDGE_SCHEMAS, WAKE_WEIGHT_COLUMN)

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


def _inverse_softplus(value: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(float(value)))))


class AeroWakeHead(nn.Module):
    """Interpretable additive aerodynamic wake term (opt-in ``model.aero_head``).

    For a follower with an incoming physical edge at frame t::

        w_t      = wake_weight_t ** k                 # = exp(-k * gap_sec / tau0)
        v2_t     = (speed_kmh_t / 100) ** 2
        corner_t = 1 - exp(-abs_curvature_t / kappa0)
        term_t   = w_t * v2_t * (a * corner_t - b * (1 - corner_t) * (1 + d * drs_t))

    ``a`` (dirty-air loss in corners), ``b`` (slipstream gain on straights), ``d``
    (extra DRS gain) and ``k`` (decay) are non-negative through softplus. The
    head returns the mean of term_t over frames where the car is active. A car
    with no incoming edge has w_t = 0 and therefore exactly zero aero term,
    which keeps clean-air predictions unchanged by construction.
    """

    def __init__(self, curvature_scale_per_m: float = 0.01):
        super().__init__()
        self.curvature_scale_per_m = float(curvature_scale_per_m)
        self.raw_corner_loss = nn.Parameter(torch.tensor(_inverse_softplus(0.05)))
        self.raw_straight_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.02)))
        self.raw_drs_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.5)))
        self.raw_decay = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))

    def coefficients(self) -> dict:
        with torch.no_grad():
            return {
                "corner_loss_a": float(F.softplus(self.raw_corner_loss)),
                "straight_gain_b": float(F.softplus(self.raw_straight_gain)),
                "drs_gain_d": float(F.softplus(self.raw_drs_gain)),
                "decay_k": float(F.softplus(self.raw_decay)),
                "curvature_scale_per_m": self.curvature_scale_per_m,
                "formula": "mean_t[w^k * (v/100)^2 * (a*corner - b*(1-corner)*(1+d*drs))]",
            }

    def forward(self, wake: torch.Tensor, aero_raw: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """wake/active: [..., W, N]; aero_raw: [..., W, N, 3] = (speed_kmh, abs_curvature, drs)."""
        a = F.softplus(self.raw_corner_loss)
        b = F.softplus(self.raw_straight_gain)
        d = F.softplus(self.raw_drs_gain)
        k = F.softplus(self.raw_decay)
        speed = torch.nan_to_num(aero_raw[..., 0], nan=0.0).clamp(min=0.0) / 100.0
        curvature = torch.nan_to_num(aero_raw[..., 1], nan=0.0).abs()
        drs = torch.nan_to_num(aero_raw[..., 2], nan=0.0).clamp(0.0, 1.0)
        corner = 1.0 - torch.exp(-curvature / self.curvature_scale_per_m)
        weight = torch.where(wake > 0, wake.clamp(min=1e-12).pow(k), torch.zeros_like(wake))
        term = weight * speed.square() * (a * corner - b * (1.0 - corner) * (1.0 + d * drs))
        active_f = active.to(term.dtype)
        count = active_f.sum(dim=-2).clamp_min(1.0)
        return (term * active_f).sum(dim=-2) / count


def _incoming_wake(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                   active: torch.Tensor) -> torch.Tensor:
    """Strongest incoming physical_v3 wake weight per node (0 without an edge)."""
    wake = torch.zeros(num_nodes, dtype=edge_attr.dtype if edge_attr is not None else torch.float32,
                       device=active.device)
    if edge_index is None or edge_attr is None or edge_index.numel() == 0:
        return wake
    src, dst = edge_index[0].to(active.device), edge_index[1].to(active.device)
    keep = active[src] & active[dst] & (src != dst)
    if not bool(keep.any()):
        return wake
    values = edge_attr[keep, WAKE_WEIGHT_COLUMN].to(device=active.device, dtype=wake.dtype).clamp(0.0, 1.0)
    return wake.scatter_reduce(0, dst[keep], values, reduce="amax", include_self=True)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HammerTimeSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for F1 Traffic Delay Prediction.
    Stage 1: Input projection (Linear)
    Stage 2: Two or three GAT/GATv2 spatial layers with dynamic edge weights
    Stage 3: Legacy Conv1d or an optional gated dilated causal residual block stack
    Stage 4: MLP head predicting delta t_sector per car, with an optional own-car correction
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
        edge_feature_schema: str = "scalar_rbf",
        gat_type: str = "gat",
        use_graph_norm: bool = False,
        drop_edge_rate: float = 0.0,
        residual: bool = False,
        typed_edge_values: bool = False,
        mask_inactive_nodes: bool = False,
        predict_future_speed: bool = False,
        future_speed_horizon_steps: int = 5,
        spatial_layers: int = 2,
        spatial_encoder: str = "gat",
        own_car_branch: bool = False,
        own_car_hidden_dim: int = 32,
        aero_head: bool = False,
    ):
        super().__init__()
        if hidden_dim <= 0 or gat_heads <= 0 or hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by gat_heads.")
        if conv1d_kernel <= 0:
            raise ValueError("conv1d_kernel must be positive.")
        if temporal_mode not in ("conv", "gated_dilated"):
            raise ValueError("temporal_mode must be 'conv' or 'gated_dilated'.")
        if gat_type not in ("gat", "gatv2"):
            raise ValueError("gat_type must be 'gat' or 'gatv2'.")
        if (not isinstance(temporal_dilations, (list, tuple)) or not temporal_dilations
                or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                       for item in temporal_dilations)):
            raise ValueError("temporal_dilations must contain positive integers.")
        if edge_feature_schema not in EDGE_FEATURES_BY_SCHEMA:
            raise ValueError(f"edge_feature_schema must be one of {list(EDGE_FEATURES_BY_SCHEMA)}.")
        if edge_dim != len(EDGE_FEATURES_BY_SCHEMA[edge_feature_schema]):
            raise ValueError("edge_dim does not match edge_feature_schema width.")
        if spatial_encoder not in ("gat", "edge_temporal"):
            raise ValueError("spatial_encoder must be 'gat' or 'edge_temporal'.")
        valid_layers = (2, 3) if spatial_encoder == "gat" else (0,)
        if (isinstance(spatial_layers, bool) or not isinstance(spatial_layers, int)
                or spatial_layers not in valid_layers):
            raise ValueError(f"spatial_layers must be one of {valid_layers} for {spatial_encoder}.")
        if not isinstance(own_car_branch, bool):
            raise ValueError("own_car_branch must be a boolean.")
        if (isinstance(own_car_hidden_dim, bool) or not isinstance(own_car_hidden_dim, int)
                or own_car_hidden_dim <= 0):
            raise ValueError("own_car_hidden_dim must be a positive integer.")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.edge_dim = edge_dim
        self.edge_feature_schema = edge_feature_schema
        self.gat_type = gat_type
        self.use_graph_norm = bool(use_graph_norm)
        self.drop_edge_rate = float(drop_edge_rate)
        self.residual = bool(residual)
        self.typed_edge_values = bool(typed_edge_values)
        self.mask_inactive_nodes = bool(mask_inactive_nodes)
        self.predict_future_speed = bool(predict_future_speed)
        self.spatial_layers = int(spatial_layers)
        self.spatial_encoder = spatial_encoder
        self.own_car_branch_enabled = bool(own_car_branch)
        self.own_car_hidden_dim = int(own_car_hidden_dim)
        if isinstance(future_speed_horizon_steps, bool) or int(future_speed_horizon_steps) <= 0:
            raise ValueError("future_speed_horizon_steps must be a positive integer.")
        self.future_speed_horizon_steps = int(future_speed_horizon_steps)
        if self.typed_edge_values and edge_feature_schema not in TYPED_EDGE_SCHEMAS:
            raise ValueError("typed_edge_values requires graph.edge_feature_schema=physical_v2 or physical_v3.")
        if self.typed_edge_values and not self.mask_inactive_nodes:
            raise ValueError("typed_edge_values requires mask_inactive_nodes=True.")
        if self.spatial_encoder == "edge_temporal" and (
                not self.typed_edge_values or edge_feature_schema not in TYPED_EDGE_SCHEMAS
                or not self.mask_inactive_nodes):
            raise ValueError("edge_temporal requires typed_edge_values, physical_v2/v3 edges, and inactive-node masking.")
        if not isinstance(aero_head, bool):
            raise ValueError("aero_head must be a boolean.")
        if aero_head and edge_feature_schema != "physical_v3":
            raise ValueError("aero_head requires graph.edge_feature_schema=physical_v3.")
        self.aero_head = AeroWakeHead() if aero_head else None
        head_dim = hidden_dim // gat_heads

        # Stage 1: Linear Input Projection
        self.proj = nn.Linear(in_dim, hidden_dim)

        # Stage 2: Spatial Message Passing (GAT or GATv2 with edge features)
        gat_cls = GATv2Conv if gat_type == "gatv2" else GATConv
        if self.spatial_encoder == "gat":
            self.gat1 = gat_cls(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=gat_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
            )
            self.gat2 = gat_cls(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=gat_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
            )
            self.gat3 = (gat_cls(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=gat_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
            ) if self.spatial_layers == 3 else None)
        else:
            self.gat1 = self.gat2 = None
            self.gat3 = None

        # Optional edge-conditioned value propagation. Each physical leader ->
        # follower pair produces two directed relation streams. Edge attributes
        # are retained verbatim for the reverse stream because relative speed
        # is already train-scaled and cannot safely be negated after scaling.
        if self.typed_edge_values:
            self.typed_value = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False)
                                             for _ in range(2))
            self.typed_edge_value = nn.ModuleList(nn.Linear(edge_dim, hidden_dim, bias=False)
                                                  for _ in range(2))
            self.typed_gate = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim)
                                            for _ in range(2))
            self.typed_norm = nn.LayerNorm(hidden_dim)
        else:
            self.typed_value = None

        if self.use_graph_norm and self.spatial_encoder == "gat":
            from torch_geometric.nn.norm import GraphNorm
            self.norm1 = GraphNorm(hidden_dim)
            self.norm2 = GraphNorm(hidden_dim)
            self.norm3 = GraphNorm(hidden_dim) if self.spatial_layers == 3 else None
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None

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
        self.future_speed_head = (nn.Linear(hidden_dim, self.future_speed_horizon_steps)
                                   if self.predict_future_speed else None)
        if self.own_car_branch_enabled:
            self.own_car_head = nn.Sequential(
                nn.Linear(in_dim, self.own_car_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.own_car_hidden_dim, 1),
            )
            nn.init.zeros_(self.own_car_head[-1].weight)
            nn.init.zeros_(self.own_car_head[-1].bias)
        else:
            self.own_car_head = None

    def _actual_node_mask(self, g, device):
        mask = getattr(g, "mask", None)
        if mask is None:
            return torch.ones(g.x.size(0), dtype=torch.bool, device=device)
        return mask.to(device=device, dtype=torch.bool)

    def _mask_graph(self, g, device):
        if not self.mask_inactive_nodes:
            return torch.ones(g.x.size(0), dtype=torch.bool, device=device)
        return self._actual_node_mask(g, device)

    def _own_car_correction(self, last_x, last_mask):
        if self.own_car_head is None:
            return last_x.new_zeros(last_x.shape[:-1])
        correction = self.own_car_head(last_x).squeeze(-1)
        return correction * last_mask.to(dtype=correction.dtype)

    def aero_coefficients(self) -> dict | None:
        return self.aero_head.coefficients() if self.aero_head is not None else None

    @staticmethod
    def _aero_raw(snapshot, device, dtype):
        raw = getattr(snapshot, "aero_raw", None)
        if raw is None:
            raise ValueError("model.aero_head requires snapshots with aero_raw [N, 3] "
                             "(speed_kmh, abs_curvature, drs); build a circuit_context_v3 / cohort_v4 dataset.")
        return raw.to(device=device, dtype=dtype)

    def _typed_messages(self, h, edge_index, edge_attr, active, normalize_empty: bool = False):
        """Aggregate typed, edge-conditioned values over physical pairs."""
        if not self.typed_edge_values:
            return h
        src, dst = edge_index
        # Only real physical pairs enter this operator: self loops are added
        # separately by the legacy attention path and never become relations.
        keep = active[src] & active[dst] & (src != dst)
        src, dst, attrs = src[keep], dst[keep], edge_attr[keep]
        if src.numel() == 0:
            return (self.typed_norm(h) * active.unsqueeze(-1)) if normalize_empty else h
        output = h.new_zeros(h.shape)
        for relation, (relation_src, relation_dst, relation_attr) in enumerate((
                (src, dst, attrs), (dst, src, attrs))):
            values = self.typed_value[relation](h[relation_src]) + self.typed_edge_value[relation](relation_attr)
            aggregate = h.new_zeros(h.shape)
            aggregate.index_add_(0, relation_dst, values)
            counts = h.new_zeros((h.size(0), 1))
            counts.index_add_(0, relation_dst, h.new_ones((relation_dst.numel(), 1)))
            aggregate = aggregate / counts.clamp_min(1.0)
            gate = torch.sigmoid(self.typed_gate[relation](torch.cat((h, aggregate), dim=-1)))
            output = output + gate * aggregate
        return self.typed_norm(h + output) * active.unsqueeze(-1)

    def _masked_graph_norm(self, values, batch, active, norm):
        """GraphNorm over active cars only, retaining inactive rows as zeros."""
        if norm is None:
            return values * active.unsqueeze(-1)
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        if graph_count == 0:
            return torch.zeros_like(values)
        weights = active.to(values.dtype).unsqueeze(-1)
        counts = values.new_zeros((graph_count, 1)).index_add_(0, batch, weights)
        sums = values.new_zeros((graph_count, values.size(-1))).index_add_(0, batch, values * weights)
        means = sums / counts.clamp_min(1.0)
        centered = values - means[batch] * norm.mean_scale
        squares = values.new_zeros((graph_count, values.size(-1))).index_add_(
            0, batch, centered.square() * weights
        )
        variance = squares / counts.clamp_min(1.0)
        normalized = centered / torch.sqrt(variance[batch] + norm.eps)
        return (normalized * norm.weight + norm.bias) * weights

    def _apply_spatial_layer(self, h, edge_index, edge_attr, active, layer, norm, batch=None):
        result = layer(h, edge_index, edge_attr)
        if norm is not None:
            if self.mask_inactive_nodes:
                graph_batch = (torch.zeros(h.size(0), dtype=torch.long, device=h.device)
                               if batch is None else batch)
                result = self._masked_graph_norm(result, graph_batch, active, norm)
            else:
                result = norm(result) if batch is None else norm(result, batch)
        result = torch.relu(result)
        return (h + result) if self.residual else result

    def _ensure_self_loops(self, num_nodes: int, edge_index: torch.Tensor,
                           edge_attr: torch.Tensor, device, position_coverage=1.0):
        """
        Adds self-loops to guaranteed graph connectivity even when active edges are 0.
        Self-loop weight is 1.0 (zero gap to self).
        """
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
        if self.edge_feature_schema == "scalar_rbf":
            loop_attr = torch.ones((num_nodes, 1), dtype=torch.float, device=device)
        elif self.edge_feature_schema == "physical_v1":
            coverage = torch.as_tensor(position_coverage, dtype=torch.float, device=device).reshape(-1)
            if coverage.numel() == 0:
                coverage = torch.zeros(1, dtype=torch.float, device=device)
            if coverage.numel() != 1:
                raise ValueError("position_coverage must be a scalar per snapshot.")
            loop_attr = torch.zeros((num_nodes, 4), dtype=torch.float, device=device)
            loop_attr[:, 0] = 1.0
            loop_attr[:, 3] = coverage[0]
        else:  # physical_v2 / physical_v3 (wake channels are 0 on self loops)
            coverage = torch.as_tensor(position_coverage, dtype=torch.float, device=device).reshape(-1)
            if coverage.numel() == 0:
                coverage = torch.zeros(1, dtype=torch.float, device=device)
            if coverage.numel() != 1:
                raise ValueError("position_coverage must be a scalar per snapshot.")
            loop_attr = torch.zeros((num_nodes, self.edge_dim), dtype=torch.float, device=device)
            loop_attr[:, 0] = 1.0
            loop_attr[:, 3] = coverage[0]

        if edge_index is None or edge_index.size(1) == 0:
            return loop_index, loop_attr
        edge_index = edge_index.to(device)
        if edge_attr is None:
            if self.edge_feature_schema == "scalar_rbf":
                edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float, device=device)
            elif self.edge_feature_schema == "physical_v1":
                edge_attr = torch.zeros((edge_index.size(1), 4), dtype=torch.float, device=device)
                edge_attr[:, 0] = 1.0
                edge_attr[:, 3] = coverage[0]
            else:
                edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), dtype=torch.float, device=device)
                edge_attr[:, 0] = 1.0
                edge_attr[:, 3] = coverage[0]
        else:
            edge_attr = edge_attr.to(device)
        if edge_attr.ndim != 2 or edge_attr.size(1) != self.edge_dim:
            raise ValueError(f"Expected edge_attr shape [E, {self.edge_dim}].")

        combined_index = torch.cat([edge_index, loop_index], dim=1)
        combined_attr = torch.cat([edge_attr, loop_attr], dim=0)
        return combined_index, combined_attr

    def forward_window(self, snapshots: list, return_aux: bool = False):
        """
        Forward pass for a single sequence of W snapshots.
        snapshots: list of W Data objects, each with x: [N, in_dim]
        Returns: [N] predicted delay per car
        """
        outs = []
        aero_wake, aero_inputs, aero_active = [], [], []
        for g in snapshots:
            device = self.proj.weight.device
            x = g.x.to(device=device, dtype=self.proj.weight.dtype)
            N = x.size(0)
            active = self._mask_graph(g, device)
            
            # 1. Project node features
            h = torch.relu(self.proj(x))
            if self.mask_inactive_nodes:
                h = h * active.unsqueeze(-1)

            # 2. Add self-loops (with optional edge dropout during training)
            raw_edge_index = getattr(g, "edge_index", None)
            raw_edge_attr = getattr(g, "edge_attr", None)
            if self.training and self.drop_edge_rate > 0.0 and raw_edge_index is not None and raw_edge_index.size(1) > 0:
                from torch_geometric.utils import dropout_edge
                raw_edge_index, edge_mask = dropout_edge(raw_edge_index, p=self.drop_edge_rate, training=True)
                if raw_edge_attr is not None:
                    raw_edge_attr = raw_edge_attr[edge_mask]
            typed_edge_index, typed_edge_attr = raw_edge_index, raw_edge_attr
            if self.aero_head is not None:
                node_active = self._actual_node_mask(g, device)
                aero_wake.append(_incoming_wake(raw_edge_index, raw_edge_attr, N, node_active))
                aero_inputs.append(self._aero_raw(g, device, self.proj.weight.dtype))
                aero_active.append(node_active)

            if self.spatial_encoder == "edge_temporal":
                if typed_edge_index is None:
                    typed_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                if typed_edge_attr is None:
                    typed_edge_attr = torch.empty((0, self.edge_dim), dtype=h.dtype, device=device)
                h = self._typed_messages(
                    h, typed_edge_index.to(device),
                    typed_edge_attr.to(device=device, dtype=h.dtype), active,
                    normalize_empty=True,
                )
            else:
                edge_idx, edge_attr = self._ensure_self_loops(
                    N, raw_edge_index, raw_edge_attr, device,
                    getattr(g, "position_coverage", 1.0),
                )
                edge_idx = edge_idx.to(device)
                edge_attr = edge_attr.to(device=device, dtype=h.dtype)
                if self.mask_inactive_nodes:
                    edge_keep = active[edge_idx[0]] & active[edge_idx[1]]
                    edge_idx, edge_attr = edge_idx[:, edge_keep], edge_attr[edge_keep]

                spatial_layers = ((self.gat1, self.norm1), (self.gat2, self.norm2))
                if self.spatial_layers == 3:
                    spatial_layers += ((self.gat3, self.norm3),)
                for layer_index, (layer, norm) in enumerate(spatial_layers):
                    h = self._apply_spatial_layer(h, edge_idx, edge_attr, active, layer, norm)
                    if layer_index == 0:
                        h = self._typed_messages(
                            h,
                            typed_edge_index.to(device) if typed_edge_index is not None else edge_idx,
                            typed_edge_attr.to(device=device, dtype=h.dtype)
                            if typed_edge_attr is not None else edge_attr,
                            active,
                        )
            # A fixed all-ones token marks an inactive historical slot without
            # adding default-path parameters or changing old checkpoint keys.
            if self.mask_inactive_nodes:
                h = torch.where(active.unsqueeze(-1), h, torch.ones_like(h))
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
        if self.own_car_head is not None:
            device = self.proj.weight.device
            last_x = snapshots[-1].x.to(device=device, dtype=self.proj.weight.dtype)
            last_mask = self._actual_node_mask(snapshots[-1], device)
            out = out + self._own_car_correction(last_x, last_mask)
        if self.aero_head is not None:
            last_mask = self._actual_node_mask(snapshots[-1], self.proj.weight.device)
            aero = self.aero_head(torch.stack(aero_wake), torch.stack(aero_inputs),
                                  torch.stack(aero_active))
            out = out + aero * last_mask.to(aero.dtype)
        if return_aux:
            aux = {}
            if self.future_speed_head is not None:
                aux["future_speed"] = self.future_speed_head(z_last)
            return out, aux
        return out

    def _forward_batched_windows(self, batch_windows, return_aux: bool = False):
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
        active_rows = []
        last_x_rows = []
        last_mask_rows = []
        typed_edges = []
        typed_attrs = []
        aero_wake_rows, aero_input_rows, aero_active_rows = [], [], []
        node_offset = 0
        for window in batch_windows:
            for time_index, snapshot in enumerate(window):
                x = snapshot.x.to(device=device, dtype=dtype)
                active = self._mask_graph(snapshot, device)
                active_rows.append(active)
                if time_index == window_size - 1:
                    last_x_rows.append(x)
                    last_mask_rows.append(self._actual_node_mask(snapshot, device))
                raw_edge_index = getattr(snapshot, "edge_index", None)
                raw_edge_attr = getattr(snapshot, "edge_attr", None)
                if self.training and self.drop_edge_rate > 0.0 and raw_edge_index is not None and raw_edge_index.size(1) > 0:
                    from torch_geometric.utils import dropout_edge
                    raw_edge_index, edge_mask = dropout_edge(raw_edge_index, p=self.drop_edge_rate, training=True)
                    if raw_edge_attr is not None:
                        raw_edge_attr = raw_edge_attr[edge_mask]
                typed_edge_index, typed_edge_attr = raw_edge_index, raw_edge_attr
                if self.aero_head is not None:
                    node_active = self._actual_node_mask(snapshot, device)
                    aero_wake_rows.append(_incoming_wake(raw_edge_index, raw_edge_attr, num_nodes, node_active))
                    aero_input_rows.append(self._aero_raw(snapshot, device, dtype))
                    aero_active_rows.append(node_active)
                if self.spatial_encoder == "edge_temporal":
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    edge_attr = torch.empty((0, self.edge_dim), dtype=dtype, device=device)
                else:
                    edge_index, edge_attr = self._ensure_self_loops(
                        num_nodes, raw_edge_index, raw_edge_attr, device,
                        getattr(snapshot, "position_coverage", 1.0),
                    )
                    if self.mask_inactive_nodes:
                        edge_keep = active[edge_index[0]] & active[edge_index[1]]
                        edge_index, edge_attr = edge_index[:, edge_keep], edge_attr[edge_keep]
                if (self.typed_edge_values and typed_edge_index is not None
                        and typed_edge_attr is not None):
                    typed_edges.append(typed_edge_index.to(device) + node_offset)
                    typed_attrs.append(typed_edge_attr.to(device=device, dtype=dtype))
                node_offset += num_nodes
                graphs.append(Data(x=x, edge_index=edge_index,
                                   edge_attr=edge_attr.to(device=device, dtype=dtype)))

        disjoint = Batch.from_data_list(graphs).to(device)
        active = torch.cat(active_rows)
        h = torch.relu(self.proj(disjoint.x))
        if self.mask_inactive_nodes:
            h = h * active.unsqueeze(-1)
        if self.spatial_encoder == "edge_temporal":
            edge_index = (torch.cat(typed_edges, dim=1) if typed_edges else
                          torch.empty((2, 0), dtype=torch.long, device=device))
            edge_attr = (torch.cat(typed_attrs) if typed_attrs else
                         torch.empty((0, self.edge_dim), dtype=dtype, device=device))
            h = self._typed_messages(h, edge_index, edge_attr, active, normalize_empty=True)
        else:
            spatial_layers = ((self.gat1, self.norm1), (self.gat2, self.norm2))
            if self.spatial_layers == 3:
                spatial_layers += ((self.gat3, self.norm3),)
            for layer_index, (layer, norm) in enumerate(spatial_layers):
                h = self._apply_spatial_layer(
                    h, disjoint.edge_index, disjoint.edge_attr, active, layer, norm, disjoint.batch
                )
                if layer_index == 0 and self.typed_edge_values and typed_edges:
                    h = self._typed_messages(h, torch.cat(typed_edges, dim=1),
                                             torch.cat(typed_attrs), active)
        if self.mask_inactive_nodes:
            h = torch.where(active.unsqueeze(-1), h, torch.ones_like(h))

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
        prediction = self.head(z_last).squeeze(-1)
        if self.own_car_head is not None:
            last_x = torch.stack(last_x_rows, dim=0)
            last_mask = torch.stack(last_mask_rows, dim=0)
            prediction = prediction + self._own_car_correction(last_x, last_mask)
        if self.aero_head is not None:
            wake = torch.stack(aero_wake_rows).reshape(batch_size, window_size, num_nodes)
            inputs = torch.stack(aero_input_rows).reshape(batch_size, window_size, num_nodes, 3)
            aero_active = torch.stack(aero_active_rows).reshape(batch_size, window_size, num_nodes)
            aero = self.aero_head(wake.to(dtype), inputs, aero_active)
            prediction = prediction + aero * aero_active[:, -1].to(aero.dtype)
        if return_aux:
            aux = {}
            if self.future_speed_head is not None:
                aux["future_speed"] = self.future_speed_head(z_last)
            return prediction, aux
        return prediction

    def forward(self, batch_windows, return_aux: bool = False):
        """
        Accepts either:
          - A single list of W Data objects -> returns [N]
          - A list of B items (each is a list of W Data objects) -> returns [B, N]
        """
        if not batch_windows:
            raise ValueError("Model input must contain at least one graph window.")
        if isinstance(batch_windows[0], (list, tuple)):
            return self._forward_batched_windows(batch_windows, return_aux=return_aux)
        return self.forward_window(batch_windows, return_aux=return_aux)


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
        edge_feature_schema=config.get("graph", {}).get("edge_feature_schema", "scalar_rbf"),
        gat_type=m_cfg.get("gat_type", "gat"),
        use_graph_norm=bool(m_cfg.get("use_graph_norm", False)),
        drop_edge_rate=float(m_cfg.get("drop_edge_rate", 0.0)),
        residual=bool(m_cfg.get("residual", False)),
        typed_edge_values=bool(m_cfg.get("typed_edge_values", False)),
        mask_inactive_nodes=bool(m_cfg.get("mask_inactive_nodes", False)),
        predict_future_speed=bool(m_cfg.get("predict_future_speed", False)),
        future_speed_horizon_steps=int(m_cfg.get("future_speed_horizon_steps", 5)),
        spatial_layers=m_cfg.get("spatial_layers", 2),
        spatial_encoder=m_cfg.get("spatial_encoder", "gat"),
        own_car_branch=bool(m_cfg.get("own_car_branch", False)),
        own_car_hidden_dim=m_cfg.get("own_car_hidden_dim", 32),
        aero_head=bool(m_cfg.get("aero_head", False)),
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
