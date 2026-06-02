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
        # top-5 sum: more readable than max on large vocabularies (32k+ tokens)
        m["confidence"] = float(np.sort(probs)[-5:].sum())

        metrics.append(m)

    return metrics


_PIPELINE_KEYS   = ["planner", "outer_12", "critic", "outer_23", "solver"]
_PIPELINE_LABELS = ["Planner", "→ outer_12 →", "Critic", "→ outer_23 →", "Solver"]


def compute_pipeline_stats(agent_vecs_per_round: list) -> list:
    """
    For each round, compute norm and cosine similarity at each pipeline stage.

    Returns a list of dicts (one per round), each with a list of stage dicts:
        {label, norm, cos_with_prev, norm_delta_pct}
    cos_with_prev and norm_delta_pct are None for the first stage.
    """
    rounds_stats = []
    for vecs in agent_vecs_per_round:
        stage_stats = []
        for i, key in enumerate(_PIPELINE_KEYS):
            v = vecs[key].float()
            norm = float(torch.norm(v))
            if i == 0:
                cos = None
                delta = None
            else:
                prev = vecs[_PIPELINE_KEYS[i - 1]].float()
                min_dim = min(len(v), len(prev))
                cos = float(F.cosine_similarity(
                    prev[:min_dim].unsqueeze(0), v[:min_dim].unsqueeze(0)
                ))
                prev_norm = float(torch.norm(prev))
                delta = (norm - prev_norm) / prev_norm * 100 if prev_norm > 0 else 0.0
            stage_stats.append({
                "label": _PIPELINE_LABELS[i],
                "norm":  round(norm, 2),
                "cos":   round(cos, 4) if cos is not None else None,
                "delta": round(delta, 1) if delta is not None else None,
            })
        rounds_stats.append(stage_stats)
    return rounds_stats


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
