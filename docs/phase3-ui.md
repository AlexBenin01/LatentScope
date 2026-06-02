# Fase 3 — UI e Deploy (Giorni 8–9)

> Carica questo file all'inizio di ogni sessione in questa fase.
> Obiettivo: Gradio funzionante con entrambe le feature, demo live su HF Spaces.

---

## Giorno 8 — UI Gradio completa

### Entry point: `src/ui/app.py`

```python
import gradio as gr
from feature1_latent import build_latent_tab
from feature2_distill import build_distill_tab

with gr.Blocks(title="RecursiveMAS Latent Visualizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # RecursiveMAS Latent Visualizer
    Prima demo pubblica che rende visibile il ragionamento interno di un MAS ricorsivo.
    Basato su [RecursiveMAS](https://arxiv.org/abs/2604.25917) — Stanford / UIUC / NVIDIA / MIT.
    """)
    
    with gr.Tabs():
        with gr.Tab("Monitor Spazio Latente"):
            build_latent_tab()
        with gr.Tab("Expert vs Learner"):
            build_distill_tab()

if __name__ == "__main__":
    demo.launch()
```

### Feature 1: `src/ui/feature1_latent.py`

```python
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def build_latent_tab():
    gr.Markdown("### Visualizza il ragionamento latente round per round")
    gr.Markdown(
        "Inserisci una domanda. Il sistema esegue 3 round ricorsivi con Sequential-Light "
        "(Planner → Critic → Solver). Il grafico mostra come cambia lo spazio latente del Solver."
    )
    
    question_input = gr.Textbox(
        label="Domanda o problema",
        placeholder="Es: Quanto fa 15 * 23? oppure: Spiega la fotosintesi",
        lines=2
    )
    run_btn = gr.Button("Esegui 3 round ricorsivi", variant="primary")
    
    with gr.Row():
        answer_out = gr.Textbox(label="Risposta finale (Round 3)", lines=4)
        metrics_plot = gr.Plot(label="Metriche spazio latente per round")
    
    with gr.Accordion("Cosa significano queste metriche?", open=False):
        gr.Markdown("""
        - **Cosine similarity**: quanto è cambiata la \"comprensione\" latente rispetto al round precedente.
          Valori vicini a 1 = il modello ha raffinato poco; valori bassi = cambiamento significativo.
        - **Confidence**: probabilità del token più probabile nell'output. Cresce se il modello diventa più sicuro.
        - **Entropia**: incertezza della distribuzione. Scende se il ragionamento converge.
        """)
    
    def run_inference(question):
        if not question.strip():
            return "Inserisci una domanda.", None
        
        # Import lazy per non bloccare il caricamento UI
        from src.inference.sequential_loop import run_recursive_loop
        from src.inference.hidden_states import extract_hidden_states
        from src.inference.metrics import compute_round_metrics
        from src.models.load_sequential import load_sequential_light
        
        # Carica modelli (in produzione: cache a livello app)
        models = load_sequential_light()
        results = run_recursive_loop(question, models)
        hidden = extract_hidden_states(results)
        metrics = compute_round_metrics(hidden, results["logits"])
        
        # Costruisci grafico
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        rounds = [1, 2, 3]
        
        sims = [m["cosine_sim"] if m["cosine_sim"] is not None else 0 for m in metrics]
        entropies = [m["entropy"] for m in metrics]
        confidences = [m["confidence"] for m in metrics]
        
        axes[0].bar(rounds, sims, color="#2E75B6")
        axes[0].set_title("Cosine Similarity\n(vs round precedente)")
        axes[0].set_ylim(0, 1)
        axes[0].set_xticks(rounds)
        
        axes[1].bar(rounds, entropies, color="#E05C2B")
        axes[1].set_title("Entropia\n(incertezza output)")
        axes[1].set_xticks(rounds)
        
        axes[2].bar(rounds, confidences, color="#70AD47")
        axes[2].set_title("Confidence\n(certezza output)")
        axes[2].set_ylim(0, 1)
        axes[2].set_xticks(rounds)
        
        plt.tight_layout()
        
        return results["answer"], fig
    
    run_btn.click(
        fn=run_inference,
        inputs=[question_input],
        outputs=[answer_out, metrics_plot]
    )
```

### Feature 2: `src/ui/feature2_distill.py`

```python
import gradio as gr
import matplotlib.pyplot as plt

def build_distill_tab():
    gr.Markdown("### Expert (9B) vs Learner (4B) — distillazione in azione")
    gr.Markdown(
        "La stessa domanda viene elaborata in parallelo da Expert e Learner. "
        "Il delta di similarity si riduce ad ogni round: il Learner converge verso l'Expert."
    )
    
    question_input = gr.Textbox(
        label="Domanda",
        placeholder="Es: Dimostra che la radice di 2 è irrazionale",
        lines=2
    )
    run_btn = gr.Button("Confronta Expert vs Learner", variant="primary")
    
    with gr.Row():
        expert_out = gr.Textbox(label="Expert — Risposta (Round 3)", lines=6)
        learner_out = gr.Textbox(label="Learner — Risposta (Round 3)", lines=6)
    
    delta_plot = gr.Plot(label="Delta Expert-Learner per round (↓ convergenza)")
    
    with gr.Row():
        expert_time = gr.Number(label="Tempo Expert (s)", precision=2)
        learner_time = gr.Number(label="Tempo Learner (s)", precision=2)
        speedup = gr.Number(label="Speedup Learner", precision=2)
    
    def run_comparison(question):
        if not question.strip():
            return "—", "—", None, 0, 0, 0
        
        import time
        from src.models.load_distillation import load_distillation
        from src.inference.sequential_loop import run_recursive_loop
        from src.inference.hidden_states import extract_hidden_states
        from src.inference.metrics import compute_expert_learner_delta
        
        models = load_distillation()
        
        t0 = time.time()
        expert_results = run_recursive_loop(question, models, role="expert")
        t_expert = time.time() - t0
        
        t0 = time.time()
        learner_results = run_recursive_loop(question, models, role="learner")
        t_learner = time.time() - t0
        
        expert_hs = extract_hidden_states(expert_results)
        learner_hs = extract_hidden_states(learner_results)
        deltas = compute_expert_learner_delta(expert_hs, learner_hs)
        
        # Grafico delta
        fig, ax = plt.subplots(figsize=(8, 4))
        rounds = [1, 2, 3]
        ax.plot(rounds, deltas, marker="o", linewidth=2.5, color="#7B5EA7", markersize=8)
        ax.fill_between(rounds, deltas, alpha=0.15, color="#7B5EA7")
        ax.set_xlabel("Round ricorsivo")
        ax.set_ylabel("Delta (1 - cosine similarity)")
        ax.set_title("Convergenza Learner → Expert\n(delta in calo = il Learner si avvicina all'Expert)")
        ax.set_xticks(rounds)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        sp = round(t_expert / t_learner, 2) if t_learner > 0 else 0
        
        return (
            expert_results["answer"],
            learner_results["answer"],
            fig,
            round(t_expert, 2),
            round(t_learner, 2),
            sp
        )
    
    run_btn.click(
        fn=run_comparison,
        inputs=[question_input],
        outputs=[expert_out, learner_out, delta_plot, expert_time, learner_time, speedup]
    )
```

### Checkpoint Giorno 8

- Entrambe le tab funzionanti con 5 domande diverse
- Nessun errore di shape o CUDA
- Grafici leggibili e con label corrette

---

## Giorno 9 — Deploy su HuggingFace Spaces

### Struttura repo HF Spaces

```
recursivemas-demo/          ← root del repo HF Space
├── app.py                  ← entry point (copia da src/ui/app.py)
├── requirements.txt
└── README.md               ← con metadati HF Spaces nell'header
```

### `requirements.txt`

```
transformers==4.47.0
torch==2.5.0
gradio==5.9.0
scipy==1.14.0
matplotlib==3.9.0
huggingface_hub==0.26.0
accelerate==1.2.0
```

> Usare versioni pinned per build riproducibili su Spaces.

### Header `README.md` per HF Spaces

```yaml
---
title: RecursiveMAS Latent Visualizer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.0
app_file: app.py
pinned: false
license: apache-2.0
---
```

### Comandi deploy

```bash
# Crea repo su HF Spaces (da terminale locale o Colab)
huggingface-cli repo create recursivemas-demo --type space --space-sdk gradio

# Clona, aggiungi file, pusha
git clone https://huggingface.co/spaces/TUO_USERNAME/recursivemas-demo
cd recursivemas-demo
# copia i file necessari
git add .
git commit -m "feat: initial deploy RecursiveMAS Latent Visualizer"
git push
```

### Checkpoint Giorno 9

- URL pubblico `https://huggingface.co/spaces/USERNAME/recursivemas-demo` accessibile
- Demo funzionante da browser diverso (non cache locale)
- Screencast da 60 secondi registrato (entrambe le feature)

---

## Comando slash deploy

File `.claude/commands/deploy.md`:

```
Prepara il deploy su HuggingFace Spaces:
1. Verifica che requirements.txt contenga versioni pinned
2. Controlla che README.md abbia l'header YAML con sdk: gradio
3. Crea un commit con messaggio "feat: deploy vX.X"
4. Mostra il comando git push da eseguire
Non fare push automaticamente — mostra solo i comandi da eseguire.
```

---

*Prossima fase: `.claude/phases/phase4-comms.md`*
