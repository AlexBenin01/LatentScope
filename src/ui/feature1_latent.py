import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from src.inference.sequential_loop import stream_recursive_loop
from src.inference.metrics import compute_round_metrics, compute_pipeline_stats

_mas = None

# ── Color palette ──────────────────────────────────────────────────────────────
_AGENT_COLORS = {
    "planner":  "#2E75B6",
    "outer_12": "#48CAE4",
    "critic":   "#E05C2B",
    "outer_23": "#F4A261",
    "solver":   "#70AD47",
}
_AGENT_LABELS = {
    "planner":  "Planner",
    "outer_12": "→ OL12 →",
    "critic":   "Critic",
    "outer_23": "→ OL23 →",
    "solver":   "Solver",
}
_KEYS   = ["planner", "outer_12", "critic", "outer_23", "solver"]
_ROUND_STYLES = ["-", "--", ":"]
_ROUND_COLORS = ["#5B9BD5", "#FF7F0E", "#2CA02C"]   # blue / orange / green per round


def set_mas(mas: dict) -> None:
    global _mas
    _mas = mas


# ── Extended PCA Journey ───────────────────────────────────────────────────────

def _pad_to(v: np.ndarray, dim: int) -> np.ndarray:
    if len(v) >= dim:
        return v[:dim]
    return np.pad(v, (0, dim - len(v)))


def _build_pca_journey(agent_vecs_per_round: list) -> plt.Figure:
    """
    Projects all 5 intermediate vectors per round (Planner/OL12/Critic/OL23/Solver)
    to 2D using PCA. Shows the full information journey across the pipeline.
    """
    n = len(agent_vecs_per_round)
    if n == 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, "In attesa del primo round...",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    MAX_DIM = 2048
    all_vecs, all_meta = [], []
    for r_idx, vecs in enumerate(agent_vecs_per_round):
        for k in _KEYS:
            all_vecs.append(_pad_to(vecs[k].numpy(), MAX_DIM))
            all_meta.append((r_idx, k))

    X   = np.stack(all_vecs)
    X_c = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    coords = X_c @ Vt[:2].T   # [n_total, 2]

    # Fix 3: variance explained for axis labels
    var = S ** 2
    pc1_pct = var[0] / var.sum() * 100
    pc2_pct = var[1] / var.sum() * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    n_agents = len(_KEYS)

    for r_idx in range(n):
        sl = slice(r_idx * n_agents, (r_idx + 1) * n_agents)
        rc = coords[sl]   # [5, 2] for this round

        # Draw intra-round path
        for i in range(n_agents - 1):
            ax.annotate("",
                xy=(rc[i+1, 0], rc[i+1, 1]),
                xytext=(rc[i, 0], rc[i, 1]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=_ROUND_COLORS[r_idx],
                    lw=1.8,
                    linestyle=_ROUND_STYLES[r_idx],
                    connectionstyle="arc3,rad=0.0",
                ))

        # Draw agent dots
        for i, k in enumerate(_KEYS):
            ax.scatter(rc[i, 0], rc[i, 1],
                       color=_AGENT_COLORS[k], s=180, zorder=5,
                       edgecolors="white", linewidths=0.8)
            if r_idx == 0:
                ax.annotate(_AGENT_LABELS[k],
                            (rc[i, 0], rc[i, 1]),
                            textcoords="offset points", xytext=(7, 5),
                            fontsize=8.5, color=_AGENT_COLORS[k], fontweight="bold")

        # Fix 1: alternate badge y-offset to avoid overlap when Planners are close
        badge_y = [-14, 0, 14][r_idx]
        ax.annotate(f" R{r_idx+1} ",
                    (rc[0, 0], rc[0, 1]),
                    textcoords="offset points", xytext=(-32, badge_y),
                    fontsize=8, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=_ROUND_COLORS[r_idx], alpha=0.85))

    # Cross-round arrows: Solver_i → Planner_{i+1} (purple)
    for r_idx in range(n - 1):
        solver_pt  = coords[r_idx * n_agents + 4]
        planner_pt = coords[(r_idx + 1) * n_agents]
        ax.annotate("",
            xy=(planner_pt[0], planner_pt[1]),
            xytext=(solver_pt[0], solver_pt[1]),
            arrowprops=dict(
                arrowstyle="->", color="#A855F7", lw=2.2,
                connectionstyle="arc3,rad=0.35",
            ))

    ax.set_title(
        "Traiettoria vettoriale completa tra agenti (PCA 2D)\n"
        "Ogni punto = vettore latente (ultimo token) — frecce colorate = round, viola = outer_31",
        fontsize=10)
    ax.set_xlabel(f"PC1 ({pc1_pct:.0f}% var. spiegata)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pc2_pct:.0f}% var. spiegata)", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Legend
    agent_handles = [
        mpatches.Patch(color=_AGENT_COLORS[k], label=_AGENT_LABELS[k])
        for k in _KEYS
    ]
    round_handles = [
        Line2D([0], [0], color=_ROUND_COLORS[i], lw=2,
               linestyle=_ROUND_STYLES[i], label=f"Round {i+1}")
        for i in range(n)
    ]
    cross_handle = Line2D([0], [0], color="#A855F7", lw=2, label="outer_31 (cross-round)")
    ax.legend(handles=agent_handles + round_handles + [cross_handle],
              loc="best", fontsize=7.5, ncol=2)

    plt.tight_layout()
    return fig


# ── Stats markdown table ────────────────────────────────────────────────────────

def _build_stats_md(pipeline_stats: list) -> str:
    if not pipeline_stats:
        return "_In attesa dei dati..._"

    lines = []
    for r_idx, stages in enumerate(pipeline_stats):
        lines.append(f"**Round {r_idx + 1}**")
        lines.append("| Stadio | Norma | Δ Norma | Cosine ← stadio prec. |")
        lines.append("|--------|------:|--------:|----------------------:|")
        for s in stages:
            delta_str = f"{s['delta']:+.0f}%" if s["delta"] is not None else "—"
            if s["cos"] is None:
                cos_str = "—"
            elif abs(s["cos"]) < 0.05:
                cos_str = f"{s['cos']:.3f} ⊥"   # near-orthogonal projection
            else:
                cos_str = f"{s['cos']:.3f}"
            lines.append(f"| {s['label']} | {s['norm']:.1f} | {delta_str} | {cos_str} |")
        lines.append("")

    return "\n".join(lines)


# ── Metric bar charts ───────────────────────────────────────────────────────────

def _build_metric_bars(hidden_states: list, metrics: list) -> plt.Figure:
    n      = len(metrics)
    rounds = list(range(1, n + 1))
    colors = [_ROUND_COLORS[i] for i in range(n)]

    sims        = [m["cosine_sim"] if m["cosine_sim"] is not None else 0.0 for m in metrics]
    entropies   = [m["entropy"]    for m in metrics]
    confidences = [m["confidence"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    sim_colors = ["#CCCCCC"] + colors[1:]
    axes[0].bar(rounds, sims, color=sim_colors)
    if sims[0] == 0.0:
        axes[0].text(1, max(sims) * 0.05 + 0.01, "N/A",
                     ha="center", fontsize=8, color="#888888", style="italic")
    axes[0].set_title("Cosine Similarity\n(Solver vs round prec.)")
    axes[0].set_ylim(0, 1); axes[0].set_xticks([1, 2, 3]); axes[0].set_xlabel("Round")

    axes[1].bar(rounds, entropies, color=colors)
    axes[1].set_title("Entropia output Solver")
    axes[1].set_xticks([1, 2, 3]); axes[1].set_xlabel("Round")

    axes[2].bar(rounds, confidences, color=colors)
    axes[2].set_title("Top-5 Confidence\n(somma 5 token più probabili)")
    axes[2].set_ylim(0, 1); axes[2].set_xticks([1, 2, 3]); axes[2].set_xlabel("Round")

    plt.tight_layout()
    return fig


# ── Gradio run generator ────────────────────────────────────────────────────────

def _run(question: str):
    """Yields (pca_chart, stats_md, metric_bars, status, answer) after each round."""
    if not question.strip():
        yield None, "_Inserisci una domanda._", None, "", ""
        return
    if _mas is None:
        yield None, "_Modelli non caricati._", None, "", ""
        return

    yield None, "_In attesa..._", None, "⏳ Round 1/3 — Planner → OL12 → Critic → OL23 → Solver...", ""

    for result in stream_recursive_loop(question, _mas):
        r       = result["round"]
        metrics = compute_round_metrics(result["hidden_states"], result["logits"])
        p_stats = compute_pipeline_stats(result["agent_vecs"])

        pca_chart    = _build_pca_journey(result["agent_vecs"])
        stats_md     = _build_stats_md(p_stats)
        metric_bars  = _build_metric_bars(result["hidden_states"], metrics)

        if result["answer"] is not None:
            status = f"✅ Round {r}/3 completato"
            yield pca_chart, stats_md, metric_bars, status, result["answer"]
        else:
            status = f"✅ Round {r}/3 completato — ⏳ Round {r+1}/3 in corso..."
            yield pca_chart, stats_md, metric_bars, status, ""


# ── Gradio tab builder ──────────────────────────────────────────────────────────

def build_latent_tab() -> None:
    gr.Markdown("### Visualizza il passaggio vettoriale tra agenti — round per round")
    gr.Markdown(
        "Ogni round: **Planner → OuterLink12 → Critic → OuterLink23 → Solver → OuterLink31**. "
        "Il grafico PCA mostra la traiettoria dei vettori latenti (ultimo token) "
        "attraverso ogni agente. La tabella mostra norma e cosine similarity ad ogni passaggio."
    )

    question_input = gr.Textbox(
        label="Domanda o problema",
        placeholder="Es: Dimostra che √2 è irrazionale  |  Risolvi: 25·3 − 6√25",
        lines=2,
    )
    run_btn = gr.Button("Esegui 3 round ricorsivi", variant="primary")
    status  = gr.Markdown(value="")

    pca_chart   = gr.Plot(label="Traiettoria vettoriale tra agenti (PCA 2D)")
    stats_panel = gr.Markdown(value="_In attesa dei dati..._",
                               label="Statistiche pipeline — norma e cosine similarity")
    metric_bars = gr.Plot(label="Metriche Solver per round")

    with gr.Accordion("Risposta finale del Planner (Round 3)", open=False):
        answer_out = gr.Textbox(label="", lines=8, show_label=False)

    with gr.Accordion("Legenda", open=False):
        gr.Markdown("""
**Grafico PCA:**
- Punti colorati per agente: Planner (blu), OL12 (ciano), Critic (arancio), OL23 (oro), Solver (verde)
- Frecce colorate per round: R1 (blu pieno), R2 (arancio tratteggiato), R3 (verde punteggiato)
- Frecce viola = outer_31: proiezione Solver→Planner tra round consecutivi

**Tabella statistiche:**
- **Norma**: intensità del vettore — indica quanto è "forte" il segnale
- **Δ Norma**: variazione percentuale rispetto allo stadio precedente
- **Cosine**: similarità direzionale con lo stadio precedente (1=stesso, 0=ortogonale)

**Barre metriche (Solver):**
- Cosine Similarity: quanto cambia la rappresentazione del Solver tra round
- Entropia: incertezza dell'output (scende = convergenza)
- Top-5 Confidence: somma delle 5 probabilità più alte
        """)

    run_btn.click(
        fn=_run,
        inputs=[question_input],
        outputs=[pca_chart, stats_panel, metric_bars, status, answer_out],
    )
