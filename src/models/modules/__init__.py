"""DIN 增强模块集合：DomainContext / PSRG-lite / PCRG-lite / TransformerFusion / MBCNet / PPNet。"""

from .domain_context import DomainContextEncoder
from .feature_slices import DEFAULT_MBCNET_GROUPS, build_feature_slices, resolve_group_slices
from .mbcnet import MBCNetHead
from .personal_context import PersonalContextEncoder
from .pcrg import PCRGLite
from .ppnet import PPNet, BranchGate, GroupWiseFiLM, resolve_ppnet_group_slices
from .psrg import PSRGLite
from .target_attention_dnn import TargetAttentionDNN
from .transformer_fusion import TransformerFusion

__all__ = [
    "DomainContextEncoder",
    "PersonalContextEncoder",
    "PSRGLite",
    "PCRGLite",
    "TargetAttentionDNN",
    "TransformerFusion",
    "MBCNetHead",
    "PPNet",
    "GroupWiseFiLM",
    "BranchGate",
    "DEFAULT_MBCNET_GROUPS",
    "build_feature_slices",
    "resolve_group_slices",
    "resolve_ppnet_group_slices",
]
