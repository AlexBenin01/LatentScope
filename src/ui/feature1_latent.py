import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference.sequential_loop import stream_recursive_loop
from src.inference.metrics import compute_round_metrics

_mas = None


def set_mas(mas: dict) -> None:
    global _mas
    _mas = mas


def _build_chart(metrics: list) -> plt.Figure:
    n      = len(metrics)
    rounds = list(range(1, n + 1))
    sims   = [m["cosine_sim"] if m["cosine_sim"] is not None else 0.0 for m in metrics]
    entropies   = [m["entropy"]    for m in metrics]
    confidences = [m["confidence"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(rounds, sims, color="#2E75B6")
    axes[0].set_title("Cosine Similarity\n(vs round precedente)")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xlabel("Round")

    axes[1].bar(rounds, entropies, color="#E05C2B")
    axes[1].set_title("Entropia\n(incertezza output)")
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xlabel("Round")

    axes[2].bar(rounds, confidences, color="#70AD47")
    axes[2].set_title("Confidence\n(certezza output)")
    axes[2].set_ylim(0, 1)
    axes[2].set_xticks([1, 2, 3])
    axes[2].set_xlabel("Round")

    plt.tight_layout()
    return fig


def _run(question: str):
    """Generator — yields (answer, chart, status) after each round."""
    if not question.strip():
        yield "Inserisci una domanda.", None, ""
        return
    if _mas is None:
        yield "Modelli non ancora caricati.", None, ""
        return

    yield "", None, "⏳ Round 1/3 — Planner → Critic → Solver..."

    for result in stream_recursive_loop(question, _mas):
        r       = result["round"]
        metrics = compute_round_metrics(result["hidden_states"], result["logits"])
        chart   = _build_chart(metrics)

        if result["answer"] is not None:
            status = f"✅ Round {r}/3 completato — risposta generata"
            yield result["answer"], chart, status
        else:
            status = f"✅ Round {r}/3 completato — ⏳ Round {r+1}/3 in corso..."
            yield "", chart, status


def build_latent_tab() -> None:
    gr.Markdown("### Visualizza il ragionamento latente round per round")
    gr.Markdown(
        "Inserisci una domanda. Il sistema esegue 3 round ricorsivi con Sequential-Light "
        "(Planner → OuterLink → Critic → OuterLink → Solver). "
        "I grafici si aggiornano in tempo reale dopo ogni round."
    )

    question_input = gr.Textbox(
        label="Domanda o problema",
        placeholder="Es: Dimostra che la radice di 2 è irrazionale",
        lines=2,
    )
    run_btn = gr.Button("Esegui 3 round ricorsivi", variant="primary")
    status  = gr.Markdown(value="")

    with gr.Row():
        answer_out   = gr.Textbox(label="Risposta finale (Round 3)", lines=8)
        metrics_plot = gr.Plot(label="Metriche spazio latente — aggiornamento live")

    with gr.Accordion("Cosa significano queste metriche?", open=False):
        gr.Markdown("""
- **Cosine similarity**: quanto è cambiata la rappresentazione latente del Solver rispetto al round precedente.
- **Confidence**: probabilità del token più probabile — cresce se il modello converge.
- **Entropia**: incertezza dell'output — scende se il ragionamento converge.
        """)

    run_btn.click(
        fn=_run,
        inputs=[question_input],
        outputs=[answer_out, metrics_plot, status],
    )
