import torch
import pytest

from src.models.din import DINModel


def _build_config(variant: str = "din_psrg_pcrg") -> dict:
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
            "user_sparse_cols": [],
            "user_dense_cols": [],
            "user_id_col": "user_id_raw",
        },
        "model": {
            "variant": variant,
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
                "query_dim": 10,  # 故意与 item_dim(=12) 不一致，测试自动投影
                "score_type": "din_mlp",
                "hidden_units": [16, 8],
                "aggregation": "mean_pool",
                "dropout": 0.0,
            },
            "fusion": {
                "mode": "concat",
                "proj_after_concat": True,
            },
            "dnn_hidden_units": [32, 16],
            "dnn_dropout": 0.0,
            "dnn_use_bn": False,
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
    }


def _build_batch(include_hist_tab: bool = False) -> dict:
    B, L = 3, 5
    batch = {
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
        "label_long_view": torch.tensor([0, 1, 0], dtype=torch.long),
    }
    if include_hist_tab:
        batch["hist_tab"] = torch.tensor(
            [[1, 2, 3, 0, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.long
        )
    return batch


def test_din_psrg_pcrg_forward_shape_and_mask_fallback():
    """
    覆盖点：
    1) hist_tab 缺失时自动回退到当前 tab 广播，不应崩溃
    2) 存在全 padding 样本时，不应产生 NaN
    3) query_dim != item_dim 时自动投影，不应报错
    """
    cfg = _build_config("din_psrg_pcrg")
    model = DINModel(cfg, _build_vocab_sizes())
    batch = _build_batch(include_hist_tab=False)

    logits = model(batch)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()

    stats = model.get_and_reset_debug_stats()
    assert "pcrg_all_pad_count" in stats
    assert stats["pcrg_all_pad_count"] >= 1


def test_hist_shape_mismatch_should_raise():
    cfg = _build_config("din_psrg_pcrg")
    model = DINModel(cfg, _build_vocab_sizes())
    batch = _build_batch(include_hist_tab=True)
    batch["hist_author_id"] = batch["hist_author_id"][:, :4]  # 制造 shape 不一致

    with pytest.raises(ValueError):
        _ = model(batch)


def test_plain_din_variant_still_works():
    cfg = _build_config("din")
    cfg["model"]["psrg"]["enabled"] = False
    cfg["model"]["pcrg"]["enabled"] = False
    model = DINModel(cfg, _build_vocab_sizes())

    batch = _build_batch(include_hist_tab=True)
    logits = model(batch)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
