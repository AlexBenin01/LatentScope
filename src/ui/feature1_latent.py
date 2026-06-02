import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference.sequential_loop import run_recursive_loop
from src.inference.hidden_states import extract_hidden_states
from src.inference.metrics import compute_round_metrics

# Populated by app.py at startup
_mas = None


def set_mas(mas: dict) -> None:
    global _mas
    _mas = mas


def _build_chart(metrics: list) -> plt.Figure:
    rounds = [1, 2, 3]
    sims   = [m["cosine_sim"] if m["cosine_sim"] is not None else 0.0 for m in metrics]
    entropies   = [m["entropy"] for m in metrics]
    confidences = [m["confidence"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(rounds, sims, color="#2E75B6")
    axes[0].set_title("Cosine Similarity\n(vs round precedente)")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(rounds)
    axes[0].set_xlabel("Round")

    axes[1].bar(rounds, entropies, color="#E05C2B")
    axes[1].set_title("Entropia\n(incertezza output)")
    axes[1].set_xticks(rounds)
    axes[1].set_xlabel("Round")

    axes[2].bar(rounds, confidences, color="#70AD47")
    axes[2].set_title("Confidence\n(certezza output)")
    axes[2].set_ylim(0, 1)
    axes[2].set_xticks(rounds)
    axes[2].set_xlabel("Round")

    plt.tight_layout()
    return fig


def _run(question: str):
    if not question.strip():
        return "Inserisci una domanda.", None
    if _mas is None:
        return "Modelli non ancora caricati.", None

    results = run_recursive_loop(question, _mas)
    hidden  = extract_hidden_states(results)
    metrics = compute_round_metrics(hidden, results["logits"])
    chart   = _build_chart(metrics)

    return results["answer"], chart


def build_latent_tab() -> None:
    gr.Markdown("### Visualizza il ragionamento latente round per round")
    gr.Markdown(
        "Inserisci una domanda. Il sistema esegue 3 round ricorsivi con Sequential-Light "
        "(Planner → OuterLink → Critic → OuterLink → Solver). "
        "Il grafico mostra come cambia lo spazio latente del Solver ad ogni round."
    )

    question_input = gr.Textbox(
        label="Domanda o problema",
        placeholder="Es: Quanto fa 15 × 23? oppure: Spiega la fotosintesi",
        lines=2,
    )
    run_btn = gr.Button("Esegui 3 round ricorsivi", variant="primary")

    with gr.Row():
        answer_out   = gr.Textbox(label="Risposta finale (Round 3)", lines=5)
        metrics_plot = gr.Plot(label="Metriche spazio latente per round")

    with gr.Accordion("Cosa significano queste metriche?", open=False):
        gr.Markdown("""
- **Cosine similarity**: quanto è cambiata la "comprensione" latente del Solver rispetto al round precedente.
  Valori vicini a 1 = raffinamento minimo; valori più bassi = cambio significativo di rappresentazione.
- **Confidence**: probabilità del token più probabile. Cresce se il modello converge verso una risposta.
- **Entropia**: incertezza della distribuzione di output. Scende se il ragionamento converge.
        """)

    run_btn.click(fn=_run, inputs=[question_input], outputs=[answer_out, metrics_plot])
