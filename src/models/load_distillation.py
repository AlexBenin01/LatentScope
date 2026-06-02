import json
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL_IDS = {
    "learner": "RecursiveMAS/Distillation-Learner-Qwen3.5-4B",
    "expert":  "RecursiveMAS/Distillation-Expert-Qwen3.5-9B",
}


def _load_model(model_id: str, role: str, **kwargs) -> AutoModelForCausalLM:
    """Same safe-load pattern as load_sequential: bypass non-standard adapter_config.json."""
    local_dir = f"/tmp/distil_{role}"
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=local_dir)

    adapter_cfg = os.path.join(local_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        if "base_model_name_or_path" not in cfg:
            os.remove(adapter_cfg)

    return AutoModelForCausalLM.from_pretrained(local_dir, **kwargs)


def load_distillation() -> dict:
    """
    Loads Expert (9B) and Learner (4B) distillation models.
    Learner is loaded first to catch OOM early.
    Combined VRAM with Sequential-Light: ~36 GB (fits on A100 40GB).
    """
    models = {}

    for role, model_id in _MODEL_IDS.items():
        print(f"Loading {role}: {model_id} ...")
        models[f"{role}_tokenizer"] = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        models[role] = _load_model(
            model_id,
            role=role,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM after {role}: {used:.1f} GB")

    return models
