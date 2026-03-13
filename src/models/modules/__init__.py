"""DIN 增强模块集合：DomainContext / PSRG-lite / PCRG-lite / TransformerFusion / MBCNet。"""

from .domain_context import DomainContextEncoder
from .feature_slices import DEFAULT_MBCNET_GROUPS, build_feature_slices, resolve_group_slices
from .mbcnet import MBCNetHead
from .pcrg import PCRGLite
from .psrg import PSRGLite
from .target_attention_dnn import TargetAttentionDNN
from .transformer_fusion import TransformerFusion

__all__ = [
    "DomainContextEncoder",
    "PSRGLite",
    "PCRGLite",
    "TargetAttentionDNN",
    "TransformerFusion",
    "MBCNetHead",
    "DEFAULT_MBCNET_GROUPS",
    "build_feature_slices",
    "resolve_group_slices",
]
