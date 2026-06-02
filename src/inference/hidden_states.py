import torch
from typing import List


def extract_hidden_states(run_results: dict) -> List[torch.Tensor]:
    """
    Returns the per-round Solver hidden states from run_recursive_loop output.
    Each tensor has shape [hidden_dim] (last token, last layer, batch squeezed).
    """
    return run_results["hidden_states"]


def verify_hidden_states_differ(hidden_states: List[torch.Tensor]) -> bool:
    """
    Sanity check: hidden states must change across rounds.
    Similarity > 0.9999 between consecutive rounds signals a bug in the loop.
    """
    if len(hidden_states) < 2:
        return True

    ok = True
    for i in range(1, len(hidden_states)):
        sim = torch.nn.functional.cosine_similarity(
            hidden_states[i - 1].unsqueeze(0),
            hidden_states[i].unsqueeze(0),
        ).item()
        print(f"  Round {i} vs {i+1} cosine similarity: {sim:.4f}")
        if sim > 0.9999:
            print(f"  WARNING: rounds {i} and {i+1} are almost identical — check the loop")
            ok = False

    if ok:
        print("  Hidden states verified: change across rounds")
    return ok
