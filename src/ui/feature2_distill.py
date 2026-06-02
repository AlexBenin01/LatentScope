import time
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference.sequential_loop import run_distillation_loop
from src.inference.metrics import compute_expert_learner_delta

# Populated by app.py at startup
_distill_models = None


def set_distill_models(models: dict) -> None:
    global _distill_models
    _distill_models = models


def _build_delta_chart(deltas: list) -> plt.Figure:
    rounds = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, deltas, marker="o", linewidth=2.5, color="#7B5EA7", markersize=8)
    ax.fill_between(rounds, deltas, alpha=0.15, color="#7B5EA7")
    ax.set_xlabel("Round di raffinamento")
    ax.set_ylabel("Delta (1 − cosine similarity)")
    ax.set_title("Convergenza Learner → Expert\n(delta in calo = il Learner si avvicina all'Expert)")
    ax.set_xticks(rounds)
    ax.set_ylim(0, max(deltas) * 1.3 if deltas else 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _run(question: str):
    if not question.strip():
        return "—", "—", None, 0.0, 0.0, 0.0
    if _distill_models is None:
        return "Modelli non ancora caricati.", "—", None, 0.0, 0.0, 0.0

    t0 = time.time()
    expert_res  = run_distillation_loop(question, _distill_models, role="expert")
    t_expert    = time.time() - t0

    t0 = time.time()
    learner_res = run_distillation_loop(question, _distill_models, role="learner")
    t_learner   = time.time() - t0

    deltas = compute_expert_learner_delta(expert_res["hidden_states"], learner_res["hidden_states"])
    chart  = _build_delta_chart(deltas)

    speedup = round(t_expert / t_learner, 2) if t_learner > 0 else 0.0

    return (
        expert_res["answers"][-1],
        learner_res["answers"][-1],
        chart,
        round(t_expert, 2),
        round(t_learner, 2),
        speedup,
    )


def build_distill_tab() -> None:
    gr.Markdown("### Expert (9B) vs Learner (4B) — distillazione in azione")
    gr.Markdown(
        "La stessa domanda viene elaborata da Expert e Learner in 3 round di raffinamento iterativo. "
        "Il grafico mostra il delta di similarity: se scende, il Learner converge verso l'Expert."
    )

    question_input = gr.Textbox(
        label="Domanda",
        placeholder="Es: Dimostra che la radice di 2 è irrazionale",
        lines=2,
    )
    run_btn = gr.Button("Confronta Expert vs Learner", variant="primary")

    with gr.Row():
        expert_out  = gr.Textbox(label="Expert — Risposta (Round 3)", lines=6)
        learner_out = gr.Textbox(label="Learner — Risposta (Round 3)", lines=6)

    delta_plot = gr.Plot(label="Delta Expert-Learner per round")

    with gr.Row():
        expert_time  = gr.Number(label="Tempo Expert (s)",    precision=2)
        learner_time = gr.Number(label="Tempo Learner (s)",   precision=2)
        speedup      = gr.Number(label="Speedup Learner (×)", precision=2)

    run_btn.click(
        fn=_run,
        inputs=[question_input],
        outputs=[expert_out, learner_out, delta_plot, expert_time, learner_time, speedup],
    )
