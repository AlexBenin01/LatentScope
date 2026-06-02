import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
import numpy as np
from typing import List, Dict, Optional


def compute_round_metrics(
    hidden_states: List[torch.Tensor],
    logits: List[torch.Tensor],
) -> List[Dict[str, Optional[float]]]:
    """
    Computes per-round metrics from Solver hidden states and logits.

    Returns a list of dicts with:
        cosine_sim:  similarity vs previous round (None for round 0)
        entropy:     output distribution entropy (lower = more focused)
        confidence:  max softmax probability (higher = more certain)
    """
    metrics = []

    for i, (hs, lgt) in enumerate(zip(hidden_states, logits)):
        m: Dict[str, Optional[float]] = {}

        m["cosine_sim"] = (
            None
            if i == 0
            else F.cosine_similarity(
                hidden_states[i - 1].unsqueeze(0),
                hs.unsqueeze(0),
            ).item()
        )

        last_logits = lgt[0, -1, :].float()
        probs = F.softmax(last_logits, dim=-1).numpy()
        m["entropy"]    = float(scipy_entropy(probs))
        m["confidence"] = float(probs.max())

        metrics.append(m)

    return metrics


def compute_expert_learner_delta(
    expert_hidden: List[torch.Tensor],
    learner_hidden: List[torch.Tensor],
) -> List[float]:
    """
    Per-round cosine distance between Expert and Learner hidden states.
    Delta = 1 - cosine_similarity.  0 = identical, 1 = orthogonal.
    Truncates to the smaller hidden dimension when models differ in size.
    """
    deltas = []
    for e, l in zip(expert_hidden, learner_hidden):
        min_dim = min(e.shape[-1], l.shape[-1])
        sim = F.cosine_similarity(
            e[:min_dim].unsqueeze(0),
            l[:min_dim].unsqueeze(0),
        ).item()
        deltas.append(round(1.0 - sim, 6))
    return deltas
