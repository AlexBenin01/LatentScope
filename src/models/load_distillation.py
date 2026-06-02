import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL_IDS = {
    "learner": "RecursiveMAS/Distillation-Learner-Qwen3.5-4B",
    "expert":  "RecursiveMAS/Distillation-Expert-Qwen3.5-9B",
}


def load_distillation() -> dict:
    """
    Loads Expert (9B) and Learner (4B) distillation models.
    Learner is loaded first to verify VRAM before loading the larger Expert.
    Combined VRAM: ~26 GB (fits on ZeroGPU / A100 40GB with bfloat16).
    """
    models = {}

    for role, model_id in _MODEL_IDS.items():
        print(f"Loading {role}: {model_id} ...")
        models[f"{role}_tokenizer"] = AutoTokenizer.from_pretrained(model_id)
        models[role] = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM after {role}: {used:.1f} GB")

    return models
