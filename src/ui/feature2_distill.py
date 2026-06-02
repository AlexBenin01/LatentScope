import time
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference.sequential_loop import stream_distillation_loop
from src.inference.metrics import compute_expert_learner_delta

_distill_models = None


def set_distill_models(models: dict) -> None:
    global _distill_models
    _distill_models = models


def _build_delta_chart(deltas: list) -> plt.Figure:
    rounds = list(range(1, len(deltas) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, deltas, marker="o", linewidth=2.5, color="#7B5EA7", markersize=8)
    ax.fill_between(rounds, deltas, alpha=0.15, color="#7B5EA7")
    ax.set_xlabel("Round di raffinamento")
    ax.set_ylabel("Delta (1 − cosine similarity)")
    ax.set_title("Convergenza Learner → Expert\n(delta in calo = il Learner si avvicina all'Expert)")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, max(max(deltas) * 1.3, 0.1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _run(question: str):
    """
    Generator — streams Expert rounds first, then Learner rounds, then delta chart.
    Yields: (expert_text, learner_text, delta_plot, status, t_expert, t_learner, speedup)
    """
    EMPTY = ("", "", None, "", 0.0, 0.0, 0.0)

    if not question.strip():
        yield ("Inserisci una domanda.", "", None, "", 0.0, 0.0, 0.0)
        return
    if _distill_models is None:
        yield ("Modelli non ancora caricati.", "", None, "", 0.0, 0.0, 0.0)
        return

    expert_hs  = []
    learner_hs = []
    expert_answers  = []
    learner_answers = []

    # --- Expert rounds ---
    t_start = time.time()
    for result in stream_distillation_loop(question, _distill_models, role="expert"):
        r = result["round"]
        expert_answers.append(result["answer"])
        expert_hs = result["hidden_states"]
        status = f"🔵 Expert — Round {r}/3 completato"
        if r < 3:
            status += f" — ⏳ Round {r+1}/3 in corso..."
        yield (
            "\n\n---\n\n".join(f"**Round {i+1}:**\n{a}" for i, a in enumerate(expert_answers)),
            "",
            None,
            status,
            0.0, 0.0, 0.0,
        )
    t_expert = time.time() - t_start

    # --- Learner rounds ---
    t_start = time.time()
    for result in stream_distillation_loop(question, _distill_models, role="learner"):
        r = result["round"]
        learner_answers.append(result["answer"])
        learner_hs = result["hidden_states"]

        deltas = compute_expert_learner_delta(expert_hs[:r], learner_hs)
        chart  = _build_delta_chart(deltas)

        status = f"🟠 Learner — Round {r}/3 completato"
        if r < 3:
            status += f" — ⏳ Round {r+1}/3 in corso..."
        else:
            status = "✅ Completato — grafico delta aggiornato"

        yield (
            "\n\n---\n\n".join(f"**Round {i+1}:**\n{a}" for i, a in enumerate(expert_answers)),
            "\n\n---\n\n".join(f"**Round {i+1}:**\n{a}" for i, a in enumerate(learner_answers)),
            chart,
            status,
            round(t_expert, 2),
            round(time.time() - t_start, 2),
            0.0,
        )

    t_learner = time.time() - t_start
    speedup   = round(t_expert / t_learner, 2) if t_learner > 0 else 0.0

    final_deltas = compute_expert_learner_delta(expert_hs, learner_hs)
    final_chart  = _build_delta_chart(final_deltas)

    yield (
        "\n\n---\n\n".join(f"**Round {i+1}:**\n{a}" for i, a in enumerate(expert_answers)),
        "\n\n---\n\n".join(f"**Round {i+1}:**\n{a}" for i, a in enumerate(learner_answers)),
        final_chart,
        "✅ Completato",
        round(t_expert, 2),
        round(t_learner, 2),
        speedup,
    )


def build_distill_tab() -> None:
    gr.Markdown("### Expert (8B) vs Learner (1.7B) — 3 round di raffinamento")
    gr.Markdown(
        "La stessa domanda viene elaborata in 3 round di raffinamento iterativo da Expert e Learner. "
        "Il grafico del delta si aggiorna live dopo ogni round del Learner."
    )

    question_input = gr.Textbox(
        label="Domanda",
        placeholder="Es: Dimostra che la radice di 2 è irrazionale",
        lines=2,
    )
    run_btn = gr.Button("Confronta Expert vs Learner", variant="primary")
    status  = gr.Markdown(value="")

    with gr.Row():
        expert_out  = gr.Markdown(label="Expert (8B) — risposte per round")
        learner_out = gr.Markdown(label="Learner (1.7B) — risposte per round")

    delta_plot = gr.Plot(label="Delta Expert-Learner per round (aggiornamento live)")

    with gr.Row():
        expert_time  = gr.Number(label="Tempo Expert (s)",    precision=2)
        learner_time = gr.Number(label="Tempo Learner (s)",   precision=2)
        speedup      = gr.Number(label="Speedup Learner (×)", precision=2)

    run_btn.click(
        fn=_run,
        inputs=[question_input],
        outputs=[expert_out, learner_out, delta_plot, status,
                 expert_time, learner_time, speedup],
    )
