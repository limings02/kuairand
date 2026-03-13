"""
特征切片与分组工具。

该文件用于把模型拼接后的 flat 向量 x=[B, D] 按语义字段切分为多个 group，
供 MBCNet 的分组交叉分支（FGC）使用。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

FeatureSlices = Dict[str, Tuple[int, int]]
WarnFn = Optional[Callable[[str], None]]

# 默认按语义分组，便于做推荐场景下的分支记忆与泛化：
# 1) interest: 当前候选相关的用户兴趣
# 2) item: 候选本身表达
# 3) user profile: 用户离散画像
# 4) user dense: 用户连续值统计
# 5) context: 场景上下文
# 6) candidate side: 候选侧额外属性
DEFAULT_MBCNET_GROUPS: List[str] = [
    "user_interest",
    "cand_repr",
    "user_profile_sparse_embs",
    "user_dense",
    "context_embs",
    "candidate_side_embs",
]


def build_feature_slices(block_dims: Dict[str, int]) -> FeatureSlices:
    """
    根据每个子块维度构建连续切片。

    Args:
        block_dims: 子块维度，要求按最终 concat 顺序给出。

    Returns:
        例如 {"user_interest": (0, 64), "cand_repr": (64, 112), ...}
    """
    slices: FeatureSlices = {}
    start = 0
    for name, dim in block_dims.items():
        end = start + int(dim)
        slices[name] = (start, end)
        start = end
    return slices


def build_uniform_group_slices(input_dim: int, num_groups: int) -> List[Tuple[str, Tuple[int, int]]]:
    """当语义切片不可用时，按均匀维度兜底分组。"""
    if input_dim <= 0:
        raise ValueError(f"input_dim 必须 > 0，当前={input_dim}")

    g = max(1, min(int(num_groups), input_dim))
    base = input_dim // g
    rem = input_dim % g

    groups: List[Tuple[str, Tuple[int, int]]] = []
    start = 0
    for idx in range(g):
        width = base + (1 if idx < rem else 0)
        end = start + width
        groups.append((f"uniform_group_{idx + 1}", (start, end)))
        start = end
    return groups


def resolve_group_slices(
    input_dim: int,
    feature_slices: Optional[FeatureSlices] = None,
    group_names: Optional[Sequence[str]] = None,
    group_dims: Optional[Sequence[int]] = None,
    fallback_num_groups: int = 4,
    warn_fn: WarnFn = None,
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    解析 MBCNet 的 group 切分策略。

    优先级：
    1) group_dims（按维度切）
    2) feature_slices + group_names（按字段名切）
    3) 均匀分组兜底（会 warning）
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim 必须 > 0，当前={input_dim}")

    if group_dims is not None:
        dims = [int(x) for x in group_dims]
        if any(d <= 0 for d in dims):
            raise ValueError(f"group_dims 必须全部为正整数，当前={dims}")
        if sum(dims) != input_dim:
            raise ValueError(
                f"group_dims 之和必须等于 input_dim，当前 sum={sum(dims)} input_dim={input_dim}"
            )
        groups: List[Tuple[str, Tuple[int, int]]] = []
        start = 0
        for i, d in enumerate(dims):
            end = start + d
            groups.append((f"group_dim_{i + 1}", (start, end)))
            start = end
        return groups

    names = list(group_names) if group_names is not None else list(DEFAULT_MBCNET_GROUPS)
    if feature_slices:
        valid_groups: List[Tuple[str, Tuple[int, int]]] = []
        missing: List[str] = []
        for name in names:
            if name not in feature_slices:
                missing.append(name)
                continue
            start, end = feature_slices[name]
            if end > start:
                valid_groups.append((name, (int(start), int(end))))

        if valid_groups:
            if missing and warn_fn is not None:
                warn_fn(f"MBCNet 分组中缺少字段切片: {missing}，将使用可用分组继续。")
            return valid_groups

    if warn_fn is not None:
        warn_fn(
            "MBCNet 未拿到可用 feature_slices/group_dims，将回退到均匀分组。"
        )
    return build_uniform_group_slices(input_dim=input_dim, num_groups=fallback_num_groups)
