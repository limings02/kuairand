"""
MBCNet 轻量三分支 Head。

目标：在不改动上游特征抽取与训练循环的前提下，把传统
    concat_features -> MLP -> logit
升级为
    concat_features -> MBCNetHead -> logit

三分支设计：
1) FGC 分支：按语义字段分组做可控显式交叉，记忆特定字段交互模式
2) Low-rank Cross 分支：低秩近似全局显式交叉，避免 O(D^2) 成本
3) Deep 分支：保留 MLP 的隐式高阶交互能力
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.models.modules.feature_slices import (
    DEFAULT_MBCNET_GROUPS,
    FeatureSlices,
    resolve_group_slices,
)

logger = logging.getLogger(__name__)


def _build_activation(name: str) -> nn.Module:
    act = str(name).lower()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    if act == "prelu":
        return nn.PReLU()
    raise ValueError(f"不支持的 activation={name}，仅支持 relu / gelu / prelu")


def _build_mlp(
    input_dim: int,
    hidden_units: Sequence[int],
    activation: str = "relu",
    dropout: float = 0.0,
    use_layernorm: bool = False,
) -> Tuple[nn.Module, int]:
    """构建轻量 MLP，并返回最终输出维度。"""
    hidden = [int(h) for h in hidden_units]
    if not hidden:
        return nn.Identity(), int(input_dim)

    layers: List[nn.Module] = []
    prev = int(input_dim)
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(_build_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers), prev


class GroupCrossLayer(nn.Module):
    """
    单层 group-wise 交叉。

    不使用 full outer product，避免 O(d^2) 中间张量。
    """

    def __init__(
        self,
        group_dim: int,
        mode: str = "cross1",
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.group_dim = int(group_dim)
        self.mode = str(mode).lower()

        if self.mode == "cross1":
            # CrossNet-1 简化版：g' = g + (w^T g) * g
            self.param = nn.Linear(self.group_dim, 1, bias=True)
        elif self.mode in {"bilinear", "gated"}:
            # bilinear: g' = g + W(g ⊙ g)
            # gated:    g' = g + sigmoid(Wg) ⊙ g
            self.param = nn.Linear(self.group_dim, self.group_dim, bias=True)
        else:
            raise ValueError(f"不支持的 FGC mode={mode}，仅支持 cross1 / bilinear / gated")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(self.group_dim) if use_layernorm else nn.Identity()

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        if self.mode == "cross1":
            scale = self.param(g)  # [B, 1]
            cross = scale * g
        elif self.mode == "bilinear":
            cross = self.param(g * g)
        else:  # gated
            gate = torch.sigmoid(self.param(g))
            cross = gate * g

        out = g + self.dropout(cross)
        return self.norm(out)


class GroupCrossBranch(nn.Module):
    """
    FGC 分支：每个 group 拥有独立参数，避免不同语义字段相互干扰。
    """

    def __init__(
        self,
        group_slices: Sequence[Tuple[str, Tuple[int, int]]],
        num_layers: int = 1,
        mode: str = "cross1",
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.group_slices = [(name, (int(s), int(e))) for name, (s, e) in group_slices]
        if not self.group_slices:
            raise ValueError("FGC 分支需要至少 1 个有效 group")

        self.group_blocks = nn.ModuleList()
        for _, (start, end) in self.group_slices:
            gdim = end - start
            if gdim <= 0:
                raise ValueError(f"非法 group 维度: ({start}, {end})")
            layers = nn.ModuleList(
                [
                    GroupCrossLayer(
                        group_dim=gdim,
                        mode=mode,
                        dropout=dropout,
                        use_layernorm=use_layernorm,
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.group_blocks.append(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for (_, (start, end)), layers in zip(self.group_slices, self.group_blocks):
            g = x[:, start:end]
            for layer in layers:
                g = layer(g)
            outs.append(g)
        return torch.cat(outs, dim=-1)


class LowRankCrossLayer(nn.Module):
    """
    低秩显式交叉层。

    公式：
        u = Ux, v = Vx
        cross = P(u ⊙ v)
        x' = x + cross

    通过 r<<D 的低秩分解近似全局交叉，复杂度约 O(B*D*r)。
    """

    def __init__(
        self,
        input_dim: int,
        rank: int = 16,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.u_proj = nn.Linear(input_dim, rank, bias=False)
        self.v_proj = nn.Linear(input_dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u_proj(x)
        v = self.v_proj(x)
        cross = self.out_proj(u * v)
        out = x + self.dropout(cross)
        return self.norm(out)


class LowRankCrossBranch(nn.Module):
    """堆叠多层低秩交叉。"""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        rank: int = 16,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LowRankCrossLayer(
                    input_dim=input_dim,
                    rank=rank,
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MBCNetHead(nn.Module):
    """MBCNet 三分支融合 Head。"""

    def __init__(
        self,
        input_dim: int,
        config: Optional[Dict[str, Any]] = None,
        feature_slices: Optional[FeatureSlices] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.config = config or {}
        self.feature_slices = feature_slices or {}

        self.enable_fgc = bool(self.config.get("enable_fgc", True))
        self.enable_lowrank_cross = bool(self.config.get("enable_lowrank_cross", True))
        self.enable_deep = bool(self.config.get("enable_deep", True))

        if not (self.enable_fgc or self.enable_lowrank_cross or self.enable_deep):
            raise ValueError("MBCNet 至少需要启用一个分支")

        # group 可按字段切，也可按维度切；若都不可用则均匀分组兜底。
        group_names = self.config.get("feature_groups", DEFAULT_MBCNET_GROUPS)
        if isinstance(group_names, str):
            group_names = [group_names]
        group_dims = self.config.get("group_dims")
        fallback_num_groups = int(self.config.get("fallback_num_groups", 4))
        self.group_slices = resolve_group_slices(
            input_dim=self.input_dim,
            feature_slices=self.feature_slices,
            group_names=group_names,
            group_dims=group_dims,
            fallback_num_groups=fallback_num_groups,
            warn_fn=lambda msg: logger.warning("[MBCNet] %s", msg),
        )

        self.fgc_cfg = self.config.get("fgc", {})
        self.lowrank_cfg = self.config.get("lowrank_cross", {})
        self.deep_cfg = self.config.get("deep", {})
        self.fusion_cfg = self.config.get("fusion", {})

        self.fgc_branch: Optional[GroupCrossBranch] = None
        self.lr_branch: Optional[LowRankCrossBranch] = None
        self.deep_branch: Optional[nn.Module] = None
        self.deep_out_dim = self.input_dim

        if self.enable_fgc:
            self.fgc_branch = GroupCrossBranch(
                group_slices=self.group_slices,
                num_layers=int(self.fgc_cfg.get("num_layers", 1)),
                mode=str(self.fgc_cfg.get("mode", "cross1")),
                dropout=float(self.fgc_cfg.get("dropout", 0.0)),
                use_layernorm=bool(self.fgc_cfg.get("layernorm", True)),
            )

        if self.enable_lowrank_cross:
            self.lr_branch = LowRankCrossBranch(
                input_dim=self.input_dim,
                num_layers=int(self.lowrank_cfg.get("num_layers", 2)),
                rank=int(self.lowrank_cfg.get("rank", 16)),
                dropout=float(self.lowrank_cfg.get("dropout", 0.0)),
                use_layernorm=bool(self.lowrank_cfg.get("layernorm", True)),
            )

        if self.enable_deep:
            deep_hidden = list(self.deep_cfg.get("hidden_units", [256, 128, 64]))
            self.deep_branch, self.deep_out_dim = _build_mlp(
                input_dim=self.input_dim,
                hidden_units=deep_hidden,
                activation=str(self.deep_cfg.get("activation", "gelu")),
                dropout=float(self.deep_cfg.get("dropout", 0.1)),
                use_layernorm=bool(self.deep_cfg.get("layernorm", False)),
            )

        self.enabled_branch_names: List[str] = []
        branch_out_dims: Dict[str, int] = {}
        if self.enable_fgc:
            self.enabled_branch_names.append("fgc")
            branch_out_dims["fgc"] = self.input_dim
        if self.enable_lowrank_cross:
            self.enabled_branch_names.append("lowrank")
            branch_out_dims["lowrank"] = self.input_dim
        if self.enable_deep:
            self.enabled_branch_names.append("deep")
            branch_out_dims["deep"] = self.deep_out_dim

        self.branch_proj_dim = int(self.fusion_cfg.get("branch_proj_dim", 128))
        self.branch_projectors = nn.ModuleDict(
            {
                name: nn.Linear(branch_out_dims[name], self.branch_proj_dim)
                for name in self.enabled_branch_names
            }
        )

        self.fusion_mode = str(self.fusion_cfg.get("mode", "concat_then_mlp")).lower()
        if self.fusion_mode not in {"concat_then_mlp", "weighted_sum"}:
            raise ValueError(
                f"不支持的 fusion.mode={self.fusion_mode}，仅支持 concat_then_mlp / weighted_sum"
            )

        if self.fusion_mode == "weighted_sum":
            # 学习三个分支（或当前启用分支）的权重。
            self.branch_weight_logits = nn.Parameter(torch.zeros(len(self.enabled_branch_names)))
        else:
            self.branch_weight_logits = None

        fusion_dropout = float(self.fusion_cfg.get("dropout", 0.1))
        final_hidden = list(self.fusion_cfg.get("final_mlp", [128, 64]))
        final_act = str(self.fusion_cfg.get("activation", "relu"))
        self.fusion_dropout = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()

        # vector head 用于：
        # 1) baseline weighted_sum
        # 2) PPNet branch gate 的加权融合
        self.vector_fusion_mlp, vector_final_dim = _build_mlp(
            input_dim=self.branch_proj_dim,
            hidden_units=final_hidden,
            activation=final_act,
            dropout=fusion_dropout,
            use_layernorm=False,
        )
        self.vector_out_linear = nn.Linear(vector_final_dim, 1)

        self.concat_fusion_mlp: Optional[nn.Module] = None
        self.concat_out_linear: Optional[nn.Linear] = None
        if self.fusion_mode == "concat_then_mlp":
            concat_input_dim = self.branch_proj_dim * len(self.enabled_branch_names)
            self.concat_fusion_mlp, concat_final_dim = _build_mlp(
                input_dim=concat_input_dim,
                hidden_units=final_hidden,
                activation=final_act,
                dropout=fusion_dropout,
                use_layernorm=False,
            )
            self.concat_out_linear = nn.Linear(concat_final_dim, 1)

        self.last_debug_stats: Dict[str, Any] = {}

    def _forward_branches(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = {}
        if self.enable_fgc and self.fgc_branch is not None:
            outs["fgc"] = self.fgc_branch(x)
        if self.enable_lowrank_cross and self.lr_branch is not None:
            outs["lowrank"] = self.lr_branch(x)
        if self.enable_deep and self.deep_branch is not None:
            outs["deep"] = self.deep_branch(x)
        return outs

    def _project_branches(self, raw_outs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            name: self.branch_projectors[name](raw_outs[name])
            for name in self.enabled_branch_names
        }

    def _collect_debug_stats(
        self,
        raw_outs: Dict[str, torch.Tensor],
        weights: Optional[torch.Tensor] = None,
        branch_gate: Optional[torch.Tensor] = None,
        fusion_source: str = "baseline",
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "mbcnet_enabled_branches": ",".join(self.enabled_branch_names),
            "mbcnet_fusion_source": fusion_source,
        }
        for name, tensor in raw_outs.items():
            norm_mean = tensor.norm(dim=-1).mean().detach().cpu().item()
            stats[f"mbcnet_{name}_norm_mean"] = float(norm_mean)

        if weights is not None:
            weight_list = [float(x) for x in weights.detach().cpu().tolist()]
            stats["mbcnet_branch_weights"] = weight_list
            for name, value in zip(self.enabled_branch_names, weight_list):
                stats[f"mbcnet_weight_{name}"] = value

        if branch_gate is not None:
            gate_mean = branch_gate.mean(dim=0).detach().cpu().tolist()
            stats["mbcnet_branch_gate_mean"] = [float(x) for x in gate_mean]
            for name, value in zip(self.enabled_branch_names, gate_mean):
                stats[f"mbcnet_gate_{name}"] = float(value)
        return stats

    def _run_vector_head(self, fused: torch.Tensor) -> torch.Tensor:
        fused = self.fusion_dropout(fused)
        fused = self.vector_fusion_mlp(fused)
        return self.vector_out_linear(fused).squeeze(-1)

    def _run_concat_head(self, fused: torch.Tensor) -> torch.Tensor:
        if self.concat_fusion_mlp is None or self.concat_out_linear is None:
            raise RuntimeError("当前 MBCNet 未构建 concat_then_mlp 融合头。")
        fused = self.fusion_dropout(fused)
        fused = self.concat_fusion_mlp(fused)
        return self.concat_out_linear(fused).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        branch_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"MBCNetHead 输入必须是 [B,D]，当前={list(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"MBCNetHead 输入维度不匹配，预期 D={self.input_dim}，实际 D={x.shape[-1]}"
            )

        raw_outs = self._forward_branches(x)
        proj_outs = self._project_branches(raw_outs)

        weights: Optional[torch.Tensor] = None
        fusion_source = "baseline"
        if branch_gate is not None:
            if branch_gate.ndim != 2:
                raise ValueError(f"branch_gate 必须是 [B,N]，当前={list(branch_gate.shape)}")
            expected_shape = (x.shape[0], len(self.enabled_branch_names))
            if tuple(branch_gate.shape) != expected_shape:
                raise ValueError(
                    f"branch_gate shape 不匹配，预期={list(expected_shape)}，实际={list(branch_gate.shape)}"
                )
            stacked = torch.stack([proj_outs[name] for name in self.enabled_branch_names], dim=1)  # [B,N,D]
            fused = (stacked * branch_gate.unsqueeze(-1)).sum(dim=1)
            logit = self._run_vector_head(fused)
            fusion_source = "ppnet_branch_gate"
        elif self.fusion_mode == "weighted_sum":
            weights = torch.softmax(self.branch_weight_logits, dim=0)  # [N]
            stacked = torch.stack([proj_outs[name] for name in self.enabled_branch_names], dim=1)  # [B,N,D]
            fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
            logit = self._run_vector_head(fused)
        else:
            fused = torch.cat([proj_outs[name] for name in self.enabled_branch_names], dim=-1)
            logit = self._run_concat_head(fused)

        self.last_debug_stats = self._collect_debug_stats(
            raw_outs,
            weights=weights,
            branch_gate=branch_gate,
            fusion_source=fusion_source,
        )
        return logit

    def get_and_reset_debug_stats(self) -> Dict[str, Any]:
        stats = self.last_debug_stats
        self.last_debug_stats = {}
        return stats
