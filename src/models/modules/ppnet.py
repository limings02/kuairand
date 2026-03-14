"""
PPNet（轻量个性化调制网络）。

为什么用 PPNet
--------------
我们希望模型最后阶段根据“当前场景 + 当前用户 + 当前活跃度代理”做轻量条件化，
从而让同一套主干表示在不同 tab / 不同用户类型 / 不同活跃度下呈现不同偏好。

这里默认实现两类极轻量调制：
1) Group-wise FiLM
   - 只对 K 个语义 group 生成 gamma/beta，而不是对 D 维全量生成参数；
   - 显存与参数量都显著更小，训练更稳定，也更好解释。
2) Branch Gate
   - 对 MBCNet 三分支（或当前启用分支）输出做 soft gate；
   - 不生成大矩阵，不引入重型 HyperNetwork。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.models.modules.feature_slices import (
    DEFAULT_MBCNET_GROUPS,
    FeatureSlices,
    build_uniform_group_slices,
)

logger = logging.getLogger(__name__)


def _build_backbone(
    input_dim: int,
    hidden_units: Sequence[int],
    dropout: float,
) -> Tuple[nn.Module, int]:
    hidden = [int(h) for h in hidden_units]
    if not hidden:
        return nn.Identity(), int(input_dim)

    layers: List[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers), prev_dim


def resolve_ppnet_group_slices(
    input_dim: int,
    feature_slices: Optional[FeatureSlices],
    group_names: Optional[Sequence[str]],
    num_groups: int,
    warn_fn: Optional[callable] = None,
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    PPNet 的 group 切分比 MBCNet 更严格：
    - 只要 `feature_slices` 缺失，或者缺少任意一个目标 group，就直接回退到均匀分组；
    - 避免 group-wise FiLM 只覆盖局部字段，导致广播调制与语义切片错位。
    """

    if input_dim <= 0:
        raise ValueError(f"PPNet input_dim 必须 > 0，当前={input_dim}")

    names = list(group_names) if group_names is not None else list(DEFAULT_MBCNET_GROUPS)
    if feature_slices:
        missing = [name for name in names if name not in feature_slices]
        if not missing:
            group_slices: List[Tuple[str, Tuple[int, int]]] = []
            total_width = 0
            for name in names:
                start, end = feature_slices[name]
                if end <= start:
                    missing.append(name)
                    continue
                group_slices.append((name, (int(start), int(end))))
                total_width += int(end - start)
            if not missing and total_width == input_dim:
                return group_slices

        if warn_fn is not None:
            warn_fn(
                f"PPNet 缺少完整 feature_slices/group_names 覆盖，"
                f"将回退到均匀分组。缺失字段: {missing or 'none'}"
            )
    elif warn_fn is not None:
        warn_fn("PPNet 未拿到 feature_slices，将回退到均匀分组。")

    return build_uniform_group_slices(input_dim=input_dim, num_groups=num_groups)


class GroupWiseFiLM(nn.Module):
    """按 group 生成 gamma/beta，并对 head 输入做广播调制。"""

    def __init__(
        self,
        context_dim: int,
        input_dim: int,
        config: Optional[Dict[str, Any]] = None,
        feature_slices: Optional[FeatureSlices] = None,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.input_dim = int(input_dim)
        self.config = config or {}
        self.dropout_prob = float(self.config.get("dropout", 0.0))
        self.output_dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()
        self.output_norm = nn.LayerNorm(self.input_dim) if bool(self.config.get("layernorm", True)) else nn.Identity()

        group_names = self.config.get("feature_groups", DEFAULT_MBCNET_GROUPS)
        if isinstance(group_names, str):
            group_names = [group_names]

        self.group_slices = resolve_ppnet_group_slices(
            input_dim=self.input_dim,
            feature_slices=feature_slices,
            group_names=group_names,
            num_groups=int(self.config.get("num_groups", 6)),
            warn_fn=lambda msg: logger.warning("[PPNet] %s", msg),
        )
        self.group_names = [name for name, _ in self.group_slices]
        self.num_groups = len(self.group_slices)

        hidden_units = list(self.config.get("hidden_units", [64, 32]))
        self.backbone, last_dim = _build_backbone(
            input_dim=self.context_dim,
            hidden_units=hidden_units,
            dropout=self.dropout_prob,
        )
        self.out_proj = nn.Linear(last_dim, 2 * self.num_groups)

        if bool(self.config.get("init_scale_zero", True)):
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        self.last_debug_stats: Dict[str, Any] = {}

    def _apply_group_film(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"PPNet GroupWiseFiLM 输入必须是 [B,D]，当前={list(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"PPNet GroupWiseFiLM 输入维度不匹配，预期 D={self.input_dim}，实际 D={x.shape[-1]}"
            )

        outputs: List[torch.Tensor] = []
        for group_idx, (_, (start, end)) in enumerate(self.group_slices):
            if end <= start:
                raise ValueError(f"PPNet group slice 非法: {(start, end)}")
            x_group = x[:, start:end]
            gamma_group = gamma[:, group_idx].unsqueeze(-1)
            beta_group = beta[:, group_idx].unsqueeze(-1)
            outputs.append((1.0 + gamma_group) * x_group + beta_group)
        return torch.cat(outputs, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        p_ctx: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        if p_ctx.ndim != 2:
            raise ValueError(f"PPNet p_ctx 必须是 [B,D_ctx]，当前={list(p_ctx.shape)}")
        if p_ctx.shape[0] != x.shape[0]:
            raise ValueError(f"PPNet batch size 不匹配: x={x.shape[0]} vs p_ctx={p_ctx.shape[0]}")

        params = self.out_proj(self.backbone(p_ctx))
        gamma, beta = torch.chunk(params, 2, dim=-1)

        x_norm_before = x.norm(dim=-1).mean()
        x_mod = self._apply_group_film(x=x, gamma=gamma, beta=beta)
        x_mod = self.output_dropout(x_mod)
        x_mod = self.output_norm(x_mod)
        x_norm_after = x_mod.norm(dim=-1).mean()

        stats = {
            "ppnet_group_names": list(self.group_names),
            "ppnet_num_groups": int(self.num_groups),
            "ppnet_gamma_mean": float(gamma.mean().detach().cpu().item()),
            "ppnet_gamma_var": float(gamma.var(unbiased=False).detach().cpu().item()),
            "ppnet_beta_mean": float(beta.mean().detach().cpu().item()),
            "ppnet_beta_var": float(beta.var(unbiased=False).detach().cpu().item()),
            "ppnet_x_norm_before_mean": float(x_norm_before.detach().cpu().item()),
            "ppnet_x_norm_after_mean": float(x_norm_after.detach().cpu().item()),
        }
        self.last_debug_stats = stats

        if return_debug:
            return x_mod, stats
        return x_mod


class BranchGate(nn.Module):
    """对 MBCNet 分支输出生成 soft gate。"""

    def __init__(
        self,
        context_dim: int,
        branch_names: Sequence[str],
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.branch_names = [str(name) for name in branch_names]
        if not self.branch_names:
            raise ValueError("BranchGate 至少需要 1 个分支名称。")

        self.config = config or {}
        dropout = float(self.config.get("dropout", 0.0))
        hidden_units = list(self.config.get("hidden_units", [64, 32]))
        self.backbone, last_dim = _build_backbone(
            input_dim=self.context_dim,
            hidden_units=hidden_units,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(last_dim, len(self.branch_names))

        # 0 初始化意味着 softmax 后接近均匀分配，更稳定。
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.last_debug_stats: Dict[str, Any] = {}

    def forward(
        self,
        p_ctx: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        logits = self.out_proj(self.backbone(p_ctx))
        gate = torch.softmax(logits, dim=-1)

        gate_mean = gate.mean(dim=0)
        stats: Dict[str, Any] = {
            "ppnet_branch_gate_mean": [float(x) for x in gate_mean.detach().cpu().tolist()],
            "ppnet_branch_names": list(self.branch_names),
        }
        for name, value in zip(self.branch_names, gate_mean.detach().cpu().tolist()):
            stats[f"ppnet_gate_{name}"] = float(value)
        self.last_debug_stats = stats

        if return_debug:
            return gate, stats
        return gate


class PPNet(nn.Module):
    """统一封装 head_input 调制与 MBCNet branch gate。"""

    def __init__(
        self,
        context_dim: int,
        input_dim: int,
        config: Optional[Dict[str, Any]] = None,
        feature_slices: Optional[FeatureSlices] = None,
        branch_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.mode = str(self.config.get("mode", "group_film")).lower()
        self.apply_to = str(self.config.get("apply_to", "head_input")).lower()
        self.context_dim = int(context_dim)
        self.input_dim = int(input_dim)

        if not self.enabled:
            self.group_film = None
            self.branch_gate = None
            self.last_debug_stats: Dict[str, Any] = {}
            return

        if self.apply_to not in {"head_input", "mbcnet_branches", "both"}:
            raise ValueError(f"不支持的 ppnet.apply_to={self.apply_to}")
        if self.mode not in {"group_film", "branch_gate"}:
            raise ValueError(f"不支持的 ppnet.mode={self.mode}")
        if self.apply_to == "head_input" and self.mode != "group_film":
            raise ValueError("ppnet.apply_to=head_input 时，ppnet.mode 必须为 group_film。")
        if self.apply_to == "mbcnet_branches" and self.mode != "branch_gate":
            raise ValueError("ppnet.apply_to=mbcnet_branches 时，ppnet.mode 必须为 branch_gate。")

        self.group_film: Optional[GroupWiseFiLM] = None
        self.branch_gate: Optional[BranchGate] = None

        if self.apply_to in {"head_input", "both"}:
            self.group_film = GroupWiseFiLM(
                context_dim=self.context_dim,
                input_dim=self.input_dim,
                config=self.config.get("group_film", {}),
                feature_slices=feature_slices,
            )

        if self.apply_to in {"mbcnet_branches", "both"}:
            branch_names = list(branch_names or [])
            if not branch_names:
                raise ValueError("ppnet.apply_to 包含 mbcnet_branches，但未提供 MBCNet 分支名称。")
            self.branch_gate = BranchGate(
                context_dim=self.context_dim,
                branch_names=branch_names,
                config=self.config.get("branch_gate", {}),
            )

        self.last_debug_stats: Dict[str, Any] = {}

    def modulate_head_input(
        self,
        x: torch.Tensor,
        p_ctx: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        if self.group_film is None:
            raise RuntimeError("当前 PPNet 未启用 head_input 调制。")
        out = self.group_film(x=x, p_ctx=p_ctx, return_debug=True)
        x_mod, stats = out
        stats = {
            "ppnet_enabled": True,
            "ppnet_apply_to": self.apply_to,
            "ppnet_mode": self.mode,
            **stats,
        }
        self.last_debug_stats.update(stats)
        if return_debug:
            return x_mod, stats
        return x_mod

    def build_branch_gate(
        self,
        p_ctx: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        if self.branch_gate is None:
            raise RuntimeError("当前 PPNet 未启用 branch gate。")
        gate, stats = self.branch_gate(p_ctx=p_ctx, return_debug=True)
        stats = {
            "ppnet_enabled": True,
            "ppnet_apply_to": self.apply_to,
            "ppnet_mode": self.mode,
            **stats,
        }
        self.last_debug_stats.update(stats)
        if return_debug:
            return gate, stats
        return gate
