import torch

from src.models.din import DINModel
from src.models.modules.feature_slices import DEFAULT_MBCNET_GROUPS
from src.models.modules.ppnet import GroupWiseFiLM, resolve_ppnet_group_slices
from src.tests.test_mbcnet_shapes import _build_batch, _build_config, _build_vocab_sizes


def _build_ppnet_config(mode: str, apply_to: str) -> dict:
    cfg = _build_config()
    cfg["model"]["ppnet"] = {
        "enabled": True,
        "mode": mode,
        "apply_to": apply_to,
        "context": {
            "use_time_features": True,
            "use_user_dense": True,
            "use_hist_len": True,
            "use_user_active_proxy": True,
            "layernorm": True,
        },
        "group_film": {
            "num_groups": 6,
            "hidden_units": [16, 8],
            "dropout": 0.0,
            "layernorm": True,
            "init_scale_zero": True,
        },
        "branch_gate": {
            "hidden_units": [16, 8],
            "dropout": 0.0,
        },
    }
    return cfg


def test_group_wise_film_broadcast_respects_feature_slices():
    feature_slices = {
        "g1": (0, 2),
        "g2": (2, 5),
        "g3": (5, 6),
    }
    film = GroupWiseFiLM(
        context_dim=4,
        input_dim=6,
        config={
            "feature_groups": ["g1", "g2", "g3"],
            "num_groups": 3,
            "hidden_units": [],
            "dropout": 0.0,
            "layernorm": False,
            "init_scale_zero": True,
        },
        feature_slices=feature_slices,
    )

    with torch.no_grad():
        # 前 3 个 bias 对应 gamma，后 3 个 bias 对应 beta。
        film.out_proj.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 3.0], dtype=torch.float32))

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        ],
        dtype=torch.float32,
    )
    p_ctx = torch.zeros(2, 4, dtype=torch.float32)
    x_mod, stats = film(x=x, p_ctx=p_ctx, return_debug=True)

    expected = torch.tensor(
        [
            [2.1, 3.2, 5.6, 6.8, 8.0, 10.8],
            [1.55, 2.65, 5.0, 6.2, 7.4, 10.15],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(x_mod, expected, atol=1e-6)
    assert stats["ppnet_num_groups"] == 3
    assert stats["ppnet_group_names"] == ["g1", "g2", "g3"]


def test_ppnet_group_slices_fallback_to_uniform_when_feature_slices_incomplete():
    resolved = resolve_ppnet_group_slices(
        input_dim=12,
        feature_slices={
            "user_interest": (0, 4),
            "cand_repr": (4, 8),
        },
        group_names=DEFAULT_MBCNET_GROUPS,
        num_groups=3,
    )
    assert [name for name, _ in resolved] == ["uniform_group_1", "uniform_group_2", "uniform_group_3"]
    assert resolved[0][1] == (0, 4)
    assert resolved[-1][1] == (8, 12)


def test_din_model_ppnet_both_reports_context_and_gate_stats():
    cfg = _build_ppnet_config(mode="group_film", apply_to="both")
    model = DINModel(cfg, _build_vocab_sizes())

    logits = model(_build_batch())
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()

    stats = model.get_and_reset_debug_stats()
    assert stats["ppnet_enabled"] is True
    assert stats["personal_context_output_dim"] > 0
    assert "ppnet_gamma_mean" in stats
    assert len(stats["ppnet_branch_gate_mean"]) == 3
    assert stats["mbcnet_fusion_source"] == "ppnet_branch_gate"
