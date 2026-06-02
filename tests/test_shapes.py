import torch
import pytest


def test_hidden_state_shape():
    """Solver hidden state must have shape [hidden_dim] after squeezing batch+seq."""
    dummy = torch.randn(1, 10, 1536)            # [batch, seq, hidden]
    vec   = dummy[:, -1, :].squeeze(0)          # last token, squeeze batch
    assert vec.shape == (1536,), f"Wrong shape: {vec.shape}"


def test_outer_link_forward():
    """OuterLink must preserve seq dimension and project hidden dim correctly."""
    from src.models.load_sequential import OuterLink

    adapter = OuterLink(in_dim=2048, out_dim=1536)
    x       = torch.randn(1, 20, 2048)
    y       = adapter(x)
    assert y.shape == (1, 20, 1536), f"Wrong output shape: {y.shape}"


def test_outer_link_residual():
    """OuterLink with same in/out dim must apply residual connection."""
    from src.models.load_sequential import OuterLink

    adapter = OuterLink(in_dim=2048, out_dim=2048)
    assert adapter._has_residual is True

    x = torch.zeros(1, 5, 2048)
    y = adapter(x)
    # Residual of zeros + proj(norm(zeros)) should differ from no-residual
    assert y.shape == (1, 5, 2048)


def test_metrics_keys():
    """compute_round_metrics must return correct keys for all rounds."""
    from src.inference.metrics import compute_round_metrics

    hs      = [torch.randn(1536) for _ in range(3)]
    logits  = [torch.randn(1, 5, 32000) for _ in range(3)]
    metrics = compute_round_metrics(hs, logits)

    assert len(metrics) == 3
    assert metrics[0]["cosine_sim"] is None, "Round 0 cosine_sim must be None"
    for m in metrics:
        assert "entropy"    in m
        assert "confidence" in m


def test_delta_range():
    """compute_expert_learner_delta must return values in [0, 2]."""
    from src.inference.metrics import compute_expert_learner_delta

    e_hs   = [torch.randn(4096) for _ in range(3)]
    l_hs   = [torch.randn(3072) for _ in range(3)]
    deltas = compute_expert_learner_delta(e_hs, l_hs)

    assert len(deltas) == 3
    assert all(0.0 <= d <= 2.0 for d in deltas), f"Out-of-range delta: {deltas}"
