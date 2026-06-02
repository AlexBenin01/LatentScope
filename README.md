# RecursiveMAS Latent Visualizer

Prima demo pubblica che rende visibile il ragionamento interno di un Multi-Agent System ricorsivo.
Basato su [RecursiveMAS](https://arxiv.org/abs/2604.25917) — Stanford · UIUC · NVIDIA · MIT (aprile 2026).

## Demo live

→ *Coming soon su HuggingFace Spaces*

## Cosa fa

**Feature 1 — Monitor spazio latente**
Inserisci una domanda. Il sistema esegue 3 round ricorsivi con Sequential-Light
(Planner → Critic → Solver collegati da OuterLinks). Il grafico mostra come cambia
lo spazio latente del Solver ad ogni round: cosine similarity, entropia, confidence.

**Feature 2 — Expert vs Learner**
Confronto side-by-side tra Expert (Qwen3.5-9B) e Learner (Qwen3.5-4B).
Il grafico mostra il delta di similarity che scende ad ogni round di raffinamento
— il Learner converge verso l'Expert.

## Setup locale

```bash
git clone https://github.com/USERNAME/LatentScope
cd LatentScope
pip install -r requirements.txt

export HF_TOKEN=hf_...   # token HuggingFace con permesso Read
python src/ui/app.py
```

Richiede GPU con almeno 10 GB VRAM per Sequential-Light, 26 GB per la feature Distillation.

## Paper originale

RecursiveMAS: Scaling Agent Collaboration through Latent-space Recursion
Yang et al., 2026 — [arXiv:2604.25917](https://arxiv.org/abs/2604.25917)

## Licenza

Apache 2.0
