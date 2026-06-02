import os
import sys

# Make src/ importable from the repo root
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

# ── ZeroGPU compatibility ──────────────────────────────────────────────────────
# Models are loaded at module level to CPU (no GPU at startup on ZeroGPU).
# @spaces.GPU on inference functions provides GPU access per-request.
try:
    import spaces
    HF_SPACES = True
except ImportError:
    HF_SPACES = False
    spaces = type("spaces", (), {"GPU": lambda fn: fn})()   # no-op decorator

# ── Model loading (runs once at startup) ──────────────────────────────────────
from src.models.load_sequential import load_sequential_light
from src.models.load_distillation import load_distillation
from src.ui.feature1_latent import build_latent_tab, set_mas, _run as _run_f1
from src.ui.feature2_distill import build_distill_tab, set_distill_models, _run as _run_f2

print("Loading Sequential-Light models...")
_mas = load_sequential_light(task="math")
set_mas(_mas)

print("Loading Distillation models...")
_distil = load_distillation()
set_distill_models(_distil)

# ── Wrap inference functions with @spaces.GPU ──────────────────────────────────
# This ensures GPU is allocated for the duration of each inference call.
# Generators are supported: GPU session stays open across all yields.
_run_f1_gpu = spaces.GPU(_run_f1)
_run_f2_gpu = spaces.GPU(_run_f2)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="RecursiveMAS Latent Visualizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# RecursiveMAS Latent Visualizer

Prima demo pubblica che rende visibile il ragionamento interno di un Multi-Agent System ricorsivo.
Basato su [RecursiveMAS](https://arxiv.org/abs/2604.25917) — Stanford · UIUC · NVIDIA · MIT (aprile 2026).
    """)

    with gr.Tabs():
        with gr.Tab("Monitor Spazio Latente"):
            build_latent_tab(run_fn=_run_f1_gpu)
        with gr.Tab("Expert vs Learner"):
            build_distill_tab(run_fn=_run_f2_gpu)


if __name__ == "__main__":
    demo.launch()
