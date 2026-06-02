import json
import os
import shutil
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_OUTERLINKS_REPO = "RecursiveMAS/Sequential-Light-Outerlinks"

_MODEL_IDS = {
    "planner": "Qwen/Qwen3-1.7B",
    "critic":  "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B",
    "solver":  "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B",
}

# OuterLink dimensions from outerlink_config.json (math task)
_OUTER_DIMS = {
    "outer_12": (2048, 2048),   # Planner -> Critic
    "outer_23": (2048, 1536),   # Critic  -> Solver
    "outer_31": (1536, 2048),   # Solver  -> Planner
}

_OUTER_FILES = {
    "math": {
        "outer_12": "Planner-Critic-Outerlink(math).pt",
        "outer_23": "Critic-Solver-Outerlink(math).pt",
        "outer_31": "Solver-Planner-Outerlink(math).pt",
    },
    "code": {
        "outer_12": "Planner-Critic-Outerlink(code).pt",
        "outer_23": "Critic-Solver-Outerlink(code).pt",
        "outer_31": "Solver-Planner-Outerlink(code).pt",
    },
}


class OuterLink(nn.Module):
    """outer_ln_res_adapter: LayerNorm -> Linear, residual when in_dim == out_dim."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self._has_residual = (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm(x))
        return h + x if self._has_residual else h


def _load_outer_link(key: str, filename: str) -> OuterLink:
    in_dim, out_dim = _OUTER_DIMS[key]
    adapter = OuterLink(in_dim, out_dim)

    path = hf_hub_download(repo_id=_OUTERLINKS_REPO, filename=filename)
    raw = torch.load(path, map_location="cpu", weights_only=True)

    # Normalize key names — handle ln.*/norm.* and linear.*/proj.* variants
    _alias = {
        "ln.weight": "norm.weight",
        "ln.bias":   "norm.bias",
        "linear.weight": "proj.weight",
        "fc.weight": "proj.weight",
    }
    state = {_alias.get(k, k): v for k, v in raw.items()}
    adapter.load_state_dict(state, strict=True)

    return adapter.to(torch.bfloat16).to(DEVICE).eval()


def _load_model(model_id: str, role: str, **kwargs) -> AutoModelForCausalLM:
    """
    Downloads the model to a role-specific temp dir and loads it, bypassing
    RecursiveMAS's non-standard adapter_config.json (only {adapter_type: ln_res_adapter})
    that breaks transformers' PEFT auto-detection.

    HF cache uses symlinks — moving a symlink doesn't hide the file from
    from_pretrained. Downloading to a fresh local_dir creates actual files
    (or local symlinks we control), so we can safely delete adapter_config.json.
    """
    local_dir = f"/tmp/mas_{role}"
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(repo_id=model_id, local_dir=local_dir)

    adapter_cfg = os.path.join(local_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        if "base_model_name_or_path" not in cfg:
            os.remove(adapter_cfg)

    return AutoModelForCausalLM.from_pretrained(local_dir, **kwargs)


def load_sequential_light(task: str = "math") -> dict:
    """
    Loads Planner, Critic, Solver + the 3 OuterLink adapters.
    Returns a flat dict with models, tokenizers, and outer adapters.
    task: "math" or "code" — selects which OuterLink weights to use.
    """
    mas = {}

    for role, model_id in _MODEL_IDS.items():
        print(f"Loading {role}: {model_id} ...")
        mas[f"{role}_tokenizer"] = AutoTokenizer.from_pretrained(model_id)
        mas[role] = _load_model(
            model_id,
            role=role,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    files = _OUTER_FILES[task]
    for key, filename in files.items():
        print(f"Loading OuterLink {key} ({filename}) ...")
        mas[key] = _load_outer_link(key, filename)

    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {used:.1f} / {total:.0f} GB")

    return mas
