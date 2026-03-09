"""
轻量 TransformerFusion 模块。

设计目标：
1) Step1: 用 Self-Attention 强化序列/多兴趣 token 内部上下文
2) Step2: 用 DIN-style target attention 强化候选与 token 的相关性
3) Step3: 再用额外 FFN 抽取更高阶、更复杂的兴趣组合

为什么默认对 Z（多兴趣 token）做 Transformer：
- Z 的长度 G 通常只有 4~8，远小于行为序列长度 L
- 在 4GB 显存场景下，Self-Attention 的二次复杂度对 G 更友好
- 因而默认 `fusion_input=interest` 可以更稳地跑通训练和 ablation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.modules.target_attention_dnn import TargetAttentionDNN, masked_mean_pool


def _build_activation(activation: str) -> nn.Module:
    act = activation.lower()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"不支持的 activation={activation}，仅支持 relu / gelu")


class LightweightTransformerEncoderLayer(nn.Module):
    """
    轻量 Transformer Encoder Layer。

    这里保留标准的：
    - MultiHeadAttention + residual + LayerNorm
    - 小型 FFN + residual + LayerNorm

    目的不是做很深的序列建模，而是在极低成本下把 token 间上下文先对齐一遍，
    让后续 target attention 看到的 token 表示更“上下文化”。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        ffn_hidden: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        ffn_hidden = ffn_hidden or max(d_model * 2, 64)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        safe_padding_mask = padding_mask
        all_pad = None
        if padding_mask is not None:
            safe_padding_mask = padding_mask.clone()
            all_pad = safe_padding_mask.all(dim=1)
            if all_pad.any():
                # 防御全 padding 样本导致 MHA softmax 出现 NaN。
                safe_padding_mask[all_pad, 0] = False

        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=safe_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        if padding_mask is not None:
            valid_mask = (~padding_mask).to(x.dtype).unsqueeze(-1)
            x = x * valid_mask
            if all_pad is not None and all_pad.any():
                x[all_pad] = 0.0
        return x


class OutputFFN(nn.Module):
    """
    Step3 的额外 FFN。

    注意 Step1 的 Transformer layer 内也有一层 FFN，但那层主要服务于 token 编码。
    这里额外再做一次 FFN，是为了对已经完成候选对齐后的 `u_target` 做更高阶兴趣抽取，
    让最终用户表示更适合送入下游 CTR DNN head。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.layernorm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm(self.net(x))


class TransformerFusion(nn.Module):
    """三步式多兴趣融合模块。"""

    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        d_model: int = 64,
        output_dim: Optional[int] = None,
        n_layers: int = 1,
        n_heads: int = 2,
        dropout: float = 0.1,
        target_att_hidden_units: Optional[List[int]] = None,
        target_att_dropout: float = 0.1,
        ffn_hidden: int = 256,
        activation: str = "gelu",
        use_layernorm: bool = True,
        use_target_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.d_model = d_model
        self.output_dim = output_dim or input_dim
        self.n_layers = n_layers
        self.use_target_attention = use_target_attention

        self.token_proj = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        self.query_proj = nn.Identity() if query_dim == d_model else nn.Linear(query_dim, d_model)

        self.encoder_layers = nn.ModuleList(
            [
                LightweightTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_hidden=max(d_model * 2, 64),
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

        self.target_attention = TargetAttentionDNN(
            item_dim=d_model,
            hidden_units=target_att_hidden_units or [64, 32],
            dropout=target_att_dropout,
            activation="prelu",
        )
        self.output_ffn = OutputFFN(
            input_dim=d_model,
            hidden_dim=ffn_hidden,
            output_dim=self.output_dim,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def _calc_token_stats(self, tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, float]:
        if mask is None:
            valid = torch.ones(tokens.shape[:2], device=tokens.device, dtype=tokens.dtype)
        else:
            valid = mask.to(tokens.dtype)

        valid = valid.unsqueeze(-1)
        denom = valid.sum().clamp_min(1.0)
        mean = (tokens * valid).sum() / denom
        var = (((tokens - mean) * valid) ** 2).sum() / denom
        return {
            "token_mean": float(mean.detach().item()),
            "token_var": float(var.detach().item()),
        }

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            query:      [B, D_q]
            tokens:     [B, N, D_in]
            token_mask: [B, N]，1=有效，0=padding；interest 模式一般可传 None
        """
        if tokens.ndim != 3:
            raise ValueError(f"TransformerFusion tokens 必须是 [B,N,D]，当前={list(tokens.shape)}")

        if token_mask is not None and token_mask.shape != tokens.shape[:2]:
            raise ValueError(
                "TransformerFusion token_mask shape 必须与 token 前两维一致，"
                f"当前 tokens={list(tokens.shape)} token_mask={list(token_mask.shape)}"
            )

        if token_mask is None:
            token_mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=tokens.dtype)

        x = self.token_proj(tokens)
        q = self.query_proj(query)

        if self.n_layers > 0:
            padding_mask = token_mask <= 0
            for layer in self.encoder_layers:
                x = layer(x, padding_mask=padding_mask)

        if self.use_target_attention:
            u_target, attn_debug = self.target_attention(
                query=q,
                keys=x,
                mask=token_mask,
                return_debug=True,
            )
            entropy = float(attn_debug.get("attn_entropy_mean", 0.0))
            all_pad_count = int(attn_debug.get("all_pad_count", 0))
        else:
            # 关闭 target attention 时退化为简单池化，用于消融 Self-Attn 本身的收益。
            u_target = masked_mean_pool(x, token_mask)
            entropy = 0.0
            all_pad_count = int((token_mask.sum(dim=-1) == 0).sum().item())

        u_fused = self.output_ffn(u_target)

        if not return_debug:
            return u_fused

        debug: Dict[str, Any] = self._calc_token_stats(x, token_mask)
        debug.update(
            {
                "target_attn_entropy_mean": entropy,
                "target_all_pad_count": all_pad_count,
            }
        )
        return u_fused, debug
