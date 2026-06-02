---
title: RecursiveMAS Latent Visualizer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.0"
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-gpu
---

# RecursiveMAS Latent Visualizer

Prima demo pubblica che rende visibile il ragionamento interno di un Multi-Agent System ricorsivo.
Basato su [RecursiveMAS](https://arxiv.org/abs/2604.25917) — Stanford · UIUC · NVIDIA · MIT (aprile 2026).

## Demo

→ Notebook Colab: scarica `notebooks/day1_setup.ipynb` e aprilo su Google Colab (runtime A100)

## Screenshots

![PCA Journey](assets/screenshot_pca_journey.png)
*Traiettoria vettoriale completa tra agenti — PCA 2D con tutti e 5 i vettori per round*

![Pipeline Stats](assets/screenshot_pipeline_stats.png)
*Statistiche pipeline: norma L2 e cosine similarity ad ogni transizione tra agenti*

## Cosa fa

### Feature 1 — Monitor spazio latente

Il sistema esegue 3 round ricorsivi con Sequential-Light (Planner → Critic → Solver, collegati da OuterLinks trainati). Visualizzazioni live aggiornate dopo ogni round:

- **PCA 2D Journey** — traiettoria di tutti e 5 i vettori latenti per ogni round. Frecce colorate per round, frecce viola per i passaggi `outer_31` cross-round. Assi con varianza spiegata (%).
- **Tabella statistiche pipeline** — norma L2 e cosine similarity ad ogni transizione. Le OuterLinks proiettano quasi ortogonalmente (cosine ≈ 0, marker `⊥`) — ogni agente opera in uno spazio semantico separato.
- **Metriche Solver per round** — cosine similarity, entropia, top-5 confidence.

### Feature 2 — Expert (8B) vs Learner (1.7B)

La stessa domanda viene elaborata in 3 round di raffinamento da Expert (Qwen3-8B) e Learner (Qwen3-1.7B). Il grafico del delta si aggiorna live dopo ogni round del Learner.

## Scoperta emergente

Le OuterLinks proiettano in modo **quasi ortogonale** tra agenti (cosine ≈ 0.0–0.06). Non preservano la direzione semantica — la trasformano completamente. Ogni agente riceve un vettore in uno spazio semantico nuovo, non una versione scalata del vettore precedente. Questo non era visualizzato numericamente in nessun lavoro precedente.

## Architettura Sequential-Light

```
Round r:
  Planner (Qwen3-1.7B)
    → outer_12 (2048→2048) ⊥
  Critic (Llama3.2-1B)
    → outer_23 (2048→1536) ⊥
  Solver (Qwen2.5-Math-1.5B)
    → outer_31 (1536→2048)
  Planner round r+1...

Round 3 → Planner genera risposta finale in italiano
```

## Setup locale

```bash
git clone https://github.com/AlexBenin01/LatentScope
cd LatentScope
pip install -r requirements.txt

python app.py   # richiede GPU ≥ 10 GB VRAM per Feature 1
```

## Test (no GPU richiesto)

```bash
pytest tests/
```

## Paper originale

RecursiveMAS: Scaling Agent Collaboration through Latent-space Recursion
Yang et al., 2026 — [arXiv:2604.25917](https://arxiv.org/abs/2604.25917)
UIUC · Stanford University · NVIDIA · MIT

## Licenza

Apache 2.0
