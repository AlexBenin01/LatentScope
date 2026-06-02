import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.inference.sequential_loop import stream_recursive_loop
from src.inference.metrics import compute_round_metrics

_mas = None


def set_mas(mas: dict) -> None:
    global _mas
    _mas = mas


def _pca_2d(hidden_states: list) -> np.ndarray:
    """Projects hidden state vectors to 2D using PCA (numpy SVD, no sklearn needed)."""
    X = np.stack([h.float().numpy() for h in hidden_states])  # [n, d]
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:2].T   # [n, 2]


def _build_charts(hidden_states: list, metrics: list) -> plt.Figure:
    """
    Returns a figure with two rows:
      Row 1: PCA 2D trajectory (main visual — mirrors Figure 7 of RecursiveMAS paper)
      Row 2: bar charts for cosine similarity, entropy, confidence
    """
    n      = len(hidden_states)
    rounds = list(range(1, n + 1))

    colors   = ["#2E75B6", "#E05C2B", "#70AD47"]
    labels_r = [f"Round {i}" for i in rounds]

    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Row 1: PCA trajectory (spans full width) ──────────────────────────
    ax_pca = fig.add_subplot(gs[0, :])

    if n >= 2:
        coords = _pca_2d(hidden_states)   # [n, 2]
        for i in range(n):
            ax_pca.scatter(coords[i, 0], coords[i, 1],
                           color=colors[i], s=220, zorder=5, label=labels_r[i])
            ax_pca.annotate(labels_r[i],
                            (coords[i, 0], coords[i, 1]),
                            textcoords="offset points", xytext=(8, 6),
                            fontsize=10, color=colors[i], fontweight="bold")
        for i in range(n - 1):
            dx = coords[i+1, 0] - coords[i, 0]
            dy = coords[i+1, 1] - coords[i, 1]
            ax_pca.annotate("",
                            xy=(coords[i+1, 0], coords[i+1, 1]),
                            xytext=(coords[i, 0], coords[i, 1]),
                            arrowprops=dict(arrowstyle="->", color="#888888",
                                            lw=1.8, connectionstyle="arc3,rad=0.1"))
        ax_pca.set_title(
            "Traiettoria dello spazio latente (PCA 2D)\n"
            "Ogni punto = rappresentazione interna del Solver dopo quel round",
            fontsize=11)
        ax_pca.set_xlabel("PC1", fontsize=9)
        ax_pca.set_ylabel("PC2", fontsize=9)
        ax_pca.legend(loc="best", fontsize=9)
        ax_pca.grid(True, alpha=0.25)
    else:
        ax_pca.text(0.5, 0.5, "In attesa di almeno 2 round...",
                    ha="center", va="center", transform=ax_pca.transAxes, fontsize=11)
        ax_pca.set_title("Traiettoria dello spazio latente (PCA 2D)")

    # ── Row 2: bar charts ──────────────────────────────────────────────────
    sims        = [m["cosine_sim"] if m["cosine_sim"] is not None else 0.0 for m in metrics]
    entropies   = [m["entropy"]    for m in metrics]
    confidences = [m["confidence"] for m in metrics]

    ax_sim  = fig.add_subplot(gs[1, 0])
    ax_ent  = fig.add_subplot(gs[1, 1])
    ax_conf = fig.add_subplot(gs[1, 2])

    bar_colors = [colors[i] for i in range(n)]

    ax_sim.bar(rounds, sims, color=bar_colors)
    ax_sim.set_title("Cosine Similarity\n(vs round precedente)")
    ax_sim.set_ylim(0, 1); ax_sim.set_xticks([1, 2, 3])
    ax_sim.set_xlabel("Round")

    ax_ent.bar(rounds, entropies, color=bar_colors)
    ax_ent.set_title("Entropia output")
    ax_ent.set_xticks([1, 2, 3]); ax_ent.set_xlabel("Round")

    ax_conf.bar(rounds, confidences, color=bar_colors)
    ax_conf.set_title("Confidence output")
    ax_conf.set_ylim(0, 1); ax_conf.set_xticks([1, 2, 3])
    ax_conf.set_xlabel("Round")

    return fig


def _run(question: str):
    """Generator — yields (chart, status, answer) after each round."""
    if not question.strip():
        yield None, "", "Inserisci una domanda."
        return
    if _mas is None:
        yield None, "", "Modelli non ancora caricati."
        return

    yield None, "⏳ Round 1/3 — Planner → Critic → Solver...", ""

    for result in stream_recursive_loop(question, _mas):
        r       = result["round"]
        metrics = compute_round_metrics(result["hidden_states"], result["logits"])
        chart   = _build_charts(result["hidden_states"], metrics)

        if result["answer"] is not None:
            status = f"✅ Round {r}/3 completato"
            yield chart, status, result["answer"]
        else:
            status = f"✅ Round {r}/3 completato — ⏳ Round {r+1}/3 in corso..."
            yield chart, status, ""


def build_latent_tab() -> None:
    gr.Markdown("### Visualizza la traiettoria nello spazio latente round per round")
    gr.Markdown(
        "Il sistema esegue 3 round ricorsivi con Sequential-Light (Planner → Critic → Solver). "
        "Il grafico PCA mostra come si sposta la rappresentazione interna del Solver nello spazio "
        "a 1536 dimensioni — aggiornamento live dopo ogni round."
    )

    question_input = gr.Textbox(
        label="Domanda o problema",
        placeholder="Es: Dimostra che √2 è irrazionale  |  Risolvi: 25·3 − 6√25",
        lines=2,
    )
    run_btn = gr.Button("Esegui 3 round ricorsivi", variant="primary")
    status  = gr.Markdown(value="")

    chart_out = gr.Plot(label="Spazio latente — aggiornamento live")

    with gr.Accordion("Risposta del Solver (Round 3)", open=False):
        answer_out = gr.Textbox(label="", lines=8, show_label=False)

    with gr.Accordion("Legenda metriche", open=False):
        gr.Markdown("""
- **PCA 2D trajectory**: proiezione dei 3 vettori nascosti (dim 1536) in 2D.
  Frecce = direzione del cambiamento; distanza = quanto è cambiata la rappresentazione.
- **Cosine Similarity**: similarità con il round precedente (0 = opposto, 1 = identico).
- **Confidence**: probabilità del token più probabile — cresce se il modello converge.
- **Entropia**: incertezza della distribuzione — scende se il ragionamento converge.
        """)

    run_btn.click(
        fn=_run,
        inputs=[question_input],
        outputs=[chart_out, status, answer_out],
    )
