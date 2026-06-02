import spaces
import gradio as gr

from src.models.load_sequential import load_sequential_light
from src.models.load_distillation import load_distillation
from src.ui.feature1_latent import build_latent_tab, set_mas
from src.ui.feature2_distill import build_distill_tab, set_distill_models

# Models are loaded once at startup (not per-request) to avoid multi-minute delays.
# @spaces.GPU ensures ZeroGPU allocates a GPU for the duration of this call.
@spaces.GPU
def _load_all_models():
    mas    = load_sequential_light(task="math")
    distil = load_distillation()
    set_mas(mas)
    set_distill_models(distil)


with gr.Blocks(title="RecursiveMAS Latent Visualizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# RecursiveMAS Latent Visualizer

Prima demo pubblica che rende visibile il ragionamento interno di un Multi-Agent System ricorsivo.
Basato su [RecursiveMAS](https://arxiv.org/abs/2604.25917) — Stanford · UIUC · NVIDIA · MIT.
    """)

    with gr.Tabs():
        with gr.Tab("Monitor Spazio Latente"):
            build_latent_tab()
        with gr.Tab("Expert vs Learner"):
            build_distill_tab()


if __name__ == "__main__":
    _load_all_models()
    demo.launch()
