import torch

from src.models.din import DINModel
from src.models.modules.mbcnet import MBCNetHead


def _build_config() -> dict:
    return {
        "fields": {
            "label_col": "label_long_view",
            "hist_seq_cols": ["hist_video_id", "hist_author_id"],
            "hist_mask_col": "hist_mask",
            "hist_len_col": "hist_len",
            "optional_hist_seq_cols": ["hist_tab", "hist_delta_t_bucket", "hist_play_ratio_bucket"],
            "cand_cols": {
                "video_id": "cand_video_id",
                "author_id": "cand_author_id",
                "video_type": "cand_video_type",
                "upload_type": "cand_upload_type",
                "duration_bucket": "cand_video_duration_bucket",
            },
            "context_sparse_cols": ["tab", "hour_of_day", "day_of_week"],
            "user_sparse_cols": ["user_active_degree", "is_video_author"],
            "user_dense_cols": ["log1p_follow_user_num", "log1p_friend_user_num"],
            "user_id_col": "user_id_raw",
        },
        "model": {
            "variant": "din_psrg_pcrg_transformer",
            "max_hist_len": 5,
            "emb_dims": {
                "video_id": 8,
                "author_id": 4,
                "small_cat": 4,
            },
            "use_hash_embedding": False,
            "large_sparse_emb_dim": 8,
            "large_sparse_threshold": 100,
            "din": {
                "att_hidden_units": [16, 8],
            },
            "hist_repr": {
                "use_hist_tab": True,
                "use_hist_delta_t_bucket": True,
                "use_hist_play_ratio_bucket": True,
            },
            "domain_context": {
                "use_hour_of_day": True,
                "use_day_of_week": True,
                "use_user_context": False,
                "hidden_units": [16],
                "output_dim": 12,
            },
            "psrg": {
                "enabled": True,
                "mode": "gated_residual",
                "hidden_units": [16],
                "layernorm": True,
                "use_hist_tab_in_psrg": True,
                "use_current_tab_always": True,
                "dropout": 0.0,
            },
            "pcrg": {
                "enabled": True,
                "num_queries": 4,
                "query_dim": 10,
                "score_type": "din_mlp",
                "hidden_units": [16, 8],
                "aggregation": "mean_pool",
                "dropout": 0.0,
            },
            "fusion": {
                "mode": "concat",
                "proj_after_concat": True,
            },
            "transformer_fusion": {
                "enabled": True,
                "fusion_input": "interest",
                "n_layers": 1,
                "n_heads": 2,
                "d_model": 12,
                "dropout": 0.0,
                "target_att_hidden_units": [16, 8],
                "target_att_dropout": 0.0,
                "use_target_attention": True,
                "ffn_hidden": 24,
                "activation": "gelu",
                "layernorm": True,
                "output_dim": 12,
                "fusion_mode": "concat",
                "proj_after_concat": True,
            },
            "dnn_hidden_units": [32, 16],
            "dnn_dropout": 0.0,
            "dnn_use_bn": False,
            "head": {
                "type": "mbcnet",
                "mbcnet": {
                    "enable_fgc": True,
                    "enable_lowrank_cross": True,
                    "enable_deep": True,
                    "fgc": {
                        "num_layers": 1,
                        "mode": "cross1",
                        "dropout": 0.0,
                        "layernorm": True,
                    },
                    "lowrank_cross": {
                        "num_layers": 2,
                        "rank": 8,
                        "dropout": 0.0,
                        "layernorm": True,
                    },
                    "deep": {
                        "hidden_units": [32, 16],
                        "activation": "gelu",
                        "dropout": 0.0,
                        "layernorm": False,
                    },
                    "fusion": {
                        "branch_proj_dim": 16,
                        "mode": "concat_then_mlp",
                        "final_mlp": [16],
                        "dropout": 0.0,
                    },
                },
            },
            "debug": {
                "print_shapes_once": False,
            },
        },
    }


def _build_vocab_sizes() -> dict:
    return {
        "video_id": 200,
        "author_id": 100,
        "video_type": 8,
        "upload_type": 8,
        "video_duration_bucket": 16,
        "tab": 20,
        "hour_of_day": 32,
        "day_of_week": 10,
        "hist_tab": 20,
        "hist_delta_t_bucket": 16,
        "hist_play_ratio_bucket": 16,
        "user_active_degree": 8,
        "is_video_author": 4,
    }


def _build_batch() -> dict:
    return {
        "hist_video_id": torch.tensor(
            [[1, 2, 3, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        "hist_author_id": torch.tensor(
            [[10, 11, 12, 0, 0], [13, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        "hist_mask": torch.tensor(
            [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        "hist_len": torch.tensor([3, 1, 0], dtype=torch.long),
        "hist_tab": torch.tensor(
            [[1, 2, 3, 0, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        "hist_delta_t_bucket": torch.tensor(
            [[1, 2, 3, 0, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        "hist_play_ratio_bucket": torch.tensor(
            [[1, 2, 2, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        "cand_video_id": torch.tensor([7, 8, 9], dtype=torch.long),
        "cand_author_id": torch.tensor([17, 18, 19], dtype=torch.long),
        "cand_video_type": torch.tensor([1, 2, 3], dtype=torch.long),
        "cand_upload_type": torch.tensor([1, 1, 2], dtype=torch.long),
        "cand_video_duration_bucket": torch.tensor([3, 4, 5], dtype=torch.long),
        "tab": torch.tensor([1, 2, 3], dtype=torch.long),
        "hour_of_day": torch.tensor([10, 11, 12], dtype=torch.long),
        "day_of_week": torch.tensor([1, 2, 3], dtype=torch.long),
        "user_active_degree": torch.tensor([1, 2, 3], dtype=torch.long),
        "is_video_author": torch.tensor([0, 1, 0], dtype=torch.long),
        "log1p_follow_user_num": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        "log1p_friend_user_num": torch.tensor([0.5, 0.6, 0.7], dtype=torch.float32),
        "label_long_view": torch.tensor([0, 1, 0], dtype=torch.long),
    }


def test_mbcnet_head_forward_shape_and_debug_stats():
    model = DINModel(_build_config(), _build_vocab_sizes())
    logits = model(_build_batch())

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()

    stats = model.get_and_reset_debug_stats()
    assert stats["head_type"] == "mbcnet"
    assert "mbcnet_fgc_norm_mean" in stats
    assert "mbcnet_lowrank_norm_mean" in stats
    assert "mbcnet_deep_norm_mean" in stats


def test_mbcnet_ablation_deep_only():
    cfg = _build_config()
    cfg["model"]["head"]["mbcnet"]["enable_fgc"] = False
    cfg["model"]["head"]["mbcnet"]["enable_lowrank_cross"] = False
    cfg["model"]["head"]["mbcnet"]["enable_deep"] = True

    model = DINModel(cfg, _build_vocab_sizes())
    logits = model(_build_batch())

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    stats = model.get_and_reset_debug_stats()
    assert "mbcnet_deep_norm_mean" in stats


def test_mbcnet_weighted_sum_has_learnable_branch_weights():
    cfg = _build_config()
    cfg["model"]["head"]["mbcnet"]["fusion"]["mode"] = "weighted_sum"
    cfg["model"]["head"]["mbcnet"]["fusion"]["final_mlp"] = [16]

    model = DINModel(cfg, _build_vocab_sizes())
    logits = model(_build_batch())

    assert logits.shape == (3,)
    stats = model.get_and_reset_debug_stats()
    weights = stats["mbcnet_branch_weights"]
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6


def test_mbcnet_fallback_uniform_groups_without_feature_slices():
    # 当没有 feature_slices 时，MBCNet 仍可回退到均匀分组，保证 shape 正常。
    cfg = {
        "enable_fgc": True,
        "enable_lowrank_cross": True,
        "enable_deep": False,
        "feature_groups": ["not_exists_a", "not_exists_b"],
        "fallback_num_groups": 3,
        "fgc": {"num_layers": 1, "mode": "cross1", "dropout": 0.0, "layernorm": True},
        "lowrank_cross": {"num_layers": 1, "rank": 4, "dropout": 0.0, "layernorm": True},
        "fusion": {"branch_proj_dim": 8, "mode": "concat_then_mlp", "final_mlp": [8], "dropout": 0.0},
    }
    head = MBCNetHead(input_dim=24, config=cfg, feature_slices=None)
    x = torch.randn(5, 24)
    logits = head(x)
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()
