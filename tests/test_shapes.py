import torch
import pytest


def test_hidden_state_shape():
    """Solver hidden state must have shape [hidden_dim] after squeezing batch+seq."""
    dummy = torch.randn(1, 10, 1536)
    vec   = dummy[:, -1, :].squeeze(0)
    assert vec.shape == (1536,), f"Wrong shape: {vec.shape}"


def test_outer_link_forward_project():
    """OuterLink (2048→1536) must output the correct target dimension."""
    from src.models.load_sequential import OuterLink
    adapter = OuterLink(in_dim=2048, out_dim=1536, mid_dim=4096)
    x = torch.randn(1, 20, 2048)
    y = adapter(x)
    assert y.shape == (1, 20, 1536), f"Wrong shape: {y.shape}"


def test_outer_link_forward_same_dim():
    """OuterLink (2048→2048) must preserve sequence dimension."""
    from src.models.load_sequential import OuterLink
    adapter = OuterLink(in_dim=2048, out_dim=2048, mid_dim=4096)
    x = torch.randn(1, 10, 2048)
    y = adapter(x)
    assert y.shape == (1, 10, 2048), f"Wrong shape: {y.shape}"


def test_outer_link_residual_proj_exists():
    """OuterLink must have a residual_proj layer connecting in_dim to out_dim."""
    from src.models.load_sequential import OuterLink
    adapter = OuterLink(in_dim=2048, out_dim=1536, mid_dim=4096)
    assert hasattr(adapter, "residual_proj")
    assert adapter.residual_proj.in_features  == 2048
    assert adapter.residual_proj.out_features == 1536


def test_metrics_keys():
    """compute_round_metrics must return correct keys for all rounds."""
    from src.inference.metrics import compute_round_metrics
    hs      = [torch.randn(1536) for _ in range(3)]
    logits  = [torch.randn(1, 5, 32000) for _ in range(3)]
    metrics = compute_round_metrics(hs, logits)
    assert len(metrics) == 3
    assert metrics[0]["cosine_sim"] is None
    for m in metrics:
        assert "entropy"    in m
        assert "confidence" in m


def test_metrics_confidence_top5():
    """confidence should be top-5 sum, so it must be >= max prob and <= 1."""
    from src.inference.metrics import compute_round_metrics
    hs     = [torch.randn(1536)]
    logits = [torch.randn(1, 5, 32000)]
    m = compute_round_metrics(hs, logits)[0]
    assert 0.0 <= m["confidence"] <= 1.0


def test_pipeline_stats_structure():
    """compute_pipeline_stats must return norms and cosines for each stage."""
    from src.inference.metrics import compute_pipeline_stats
    vecs = {k: torch.randn(2048 if k not in ("outer_23", "solver") else 1536)
            for k in ["planner", "outer_12", "critic", "outer_23", "solver"]}
    stats = compute_pipeline_stats([vecs])
    assert len(stats) == 1
    assert len(stats[0]) == 5
    assert stats[0][0]["cos"]   is None   # first stage has no previous
    assert stats[0][0]["delta"] is None
    for s in stats[0][1:]:
        assert s["cos"]   is not None
        assert s["delta"] is not None
        assert s["norm"]  > 0


def test_delta_range():
    """compute_expert_learner_delta must return values in [0, 2]."""
    from src.inference.metrics import compute_expert_learner_delta
    e_hs   = [torch.randn(4096) for _ in range(3)]
    l_hs   = [torch.randn(3072) for _ in range(3)]
    deltas = compute_expert_learner_delta(e_hs, l_hs)
    assert len(deltas) == 3
    assert all(0.0 <= d <= 2.0 for d in deltas)
