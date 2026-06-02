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
---

# RecursiveMAS Latent Visualizer

Prima demo pubblica che rende visibile il ragionamento interno di un Multi-Agent System ricorsivo.
Basato su [RecursiveMAS](https://arxiv.org/abs/2604.25917) — Stanford · UIUC · NVIDIA · MIT (aprile 2026).

## Demo live

→ *Coming soon su HuggingFace Spaces*

## Cosa fa

### Feature 1 — Monitor spazio latente

Inserisci una domanda. Il sistema esegue 3 round ricorsivi con Sequential-Light
(Planner → OuterLink → Critic → OuterLink → Solver).

**Visualizzazioni live, aggiornate dopo ogni round:**

- **PCA 2D Journey** — traiettoria di tutti e 5 i vettori latenti (Planner, OL12, Critic, OL23, Solver) per ogni round. Frecce colorate per round, frecce viola per i passaggi `outer_31` cross-round
- **Tabella statistiche pipeline** — norma L2 e cosine similarity ad ogni transizione tra agenti. Le OuterLinks proiettano quasi ortogonalmente (cosine ≈ 0), visibile con il marker `⊥`
- **Metriche Solver** — cosine similarity tra round, entropia e top-5 confidence

### Feature 2 — Expert (8B) vs Learner (1.7B)

La stessa domanda viene elaborata in 3 round di raffinamento da Expert (Qwen3-8B) e Learner (Qwen3-1.7B).
Il grafico del delta di similarity si aggiorna live dopo ogni round del Learner.

## Architettura

```
Round 1:
  Planner (Qwen3-1.7B) → outer_12 → Critic (Llama3.2-1B) → outer_23 → Solver (Qwen2.5-Math-1.5B)
                                                                              ↓
                                                                          outer_31
                                                                              ↓
Round 2:                                                               → Planner ...
```

Il Planner genera la risposta finale in italiano usando il contesto latente del Solver proiettato via `outer_31`.

## Setup locale

```bash
git clone https://github.com/AlexBenin01/LatentScope
cd LatentScope
pip install -r requirements.txt

# Serve token HuggingFace per scaricare i modelli RecursiveMAS
export HF_TOKEN=hf_...
python app.py
```

Richiede GPU con almeno 10 GB VRAM per Feature 1, ~36 GB totali per Feature 1 + Feature 2.

## Test

```bash
pytest tests/
```

I test non richiedono GPU — usano tensori casuali per verificare shapes e metriche.

## Paper originale

RecursiveMAS: Scaling Agent Collaboration through Latent-space Recursion
Yang et al., 2026 — [arXiv:2604.25917](https://arxiv.org/abs/2604.25917)
UIUC · Stanford University · NVIDIA · MIT

## Licenza

MIT
