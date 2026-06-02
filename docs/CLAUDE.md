# RecursiveMAS Latent Visualizer — Piano di Progetto

> Prima demo pubblica che visualizza lo spazio latente di un Multi-Agent System ricorsivo.
> Basato su [RecursiveMAS](https://arxiv.org/abs/2604.25917) (Stanford / UIUC / NVIDIA / MIT, aprile 2026).

---

## Contesto del progetto

Questo file è il `CLAUDE.md` radice del progetto. Claude Code lo legge all'avvio di ogni sessione.
Mantienilo sotto 200 righe — i dettagli tecnici vivono nei file di fase.

### Cosa stiamo costruendo

Un'app Gradio con due feature originali, deployata su HuggingFace Spaces:

1. **Monitor spazio latente** — 3 round ricorsivi Sequential-Light con grafico cosine similarity / entropia per round
2. **Expert vs Learner** — confronto side-by-side Qwen3.5-9B vs Qwen3.5-4B con progress bar delta

### Perché è originale

- Usa modelli **open-weight con RecursiveLink già trainato** (impossibile con API closed-source)
- Rende visibile il ragionamento latente inter-round — nessuno l'ha ancora fatto
- Paper pubblicato 5 giorni fa: finestra di opportunità reale

### Stack definitivo

| Layer | Tool |
|---|---|
| Ambiente | Google Colab Pro (A100 40GB) |
| Modelli | `HuggingFace RecursiveMAS/*` |
| Inference | `transformers` + `torch` |
| Hidden states | `output_hidden_states=True` + forward hooks |
| Metriche | `torch.nn.functional` + `scipy` |
| UI | `gradio` |
| Hosting | HuggingFace Spaces (gratuito) |
| Versioning | GitHub (repo pubblica) |

### Regole operative per Claude Code

- Ogni sessione: leggi questo file + il file di fase corrente in `.claude/phases/`
- Diff massimo per sessione: ~150 righe — se superi, spezza in sotto-task
- Dopo ogni task completato: aggiorna `STATUS.md` con il progresso
- Non committare mai su `main` direttamente — usa branch `feat/giorno-N`
- Se la sessione si allunga: salva il piano su `PROGRESS.md` e usa `/clear`

---

## Struttura del progetto

```
recursivemas-demo/
├── CLAUDE.md                  ← questo file (radice)
├── STATUS.md                  ← stato corrente del progetto
├── PROGRESS.md                ← dump piano/progresso prima di /clear
├── .claude/
│   ├── phases/
│   │   ├── phase1-setup.md
│   │   ├── phase2-core.md
│   │   ├── phase3-ui.md
│   │   └── phase4-comms.md
│   └── commands/
│       ├── run-tests.md
│       ├── checkpoint.md
│       └── deploy.md
├── src/
│   ├── inference/
│   │   ├── sequential_loop.py     ← loop ricorsivo 3 round
│   │   ├── hidden_states.py       ← estrazione vettori latenti
│   │   └── metrics.py             ← cosine sim, entropia, confidence
│   ├── models/
│   │   ├── load_sequential.py     ← caricamento Sequential-Light
│   │   └── load_distillation.py   ← caricamento Expert + Learner
│   └── ui/
│       ├── app.py                 ← entry point Gradio
│       ├── feature1_latent.py     ← UI Monitor spazio latente
│       └── feature2_distill.py    ← UI Expert vs Learner
├── notebooks/
│   ├── day1_setup.ipynb
│   ├── day4_hidden_states.ipynb   ← notebook critico
│   └── day7_distillation.ipynb
├── tests/
│   ├── test_inference.py
│   ├── test_metrics.py
│   └── test_shapes.py             ← verifica shape tensori
├── requirements.txt
└── README.md
```

---

## Fase corrente

Vedi `.claude/phases/phase1-setup.md` per il dettaglio operativo della fase attiva.
Aggiorna questa riga quando cambi fase.

**Fase attiva: FASE 1 — Setup** (Giorni 1–2)

---

## KPI di progetto

| KPI | Target minimo | Target ambizioso |
|---|---|---|
| Demo live HF Spaces | ✅ SI | 50+ run nella prima settimana |
| Impression LinkedIn | 2.000 | 10.000+ |
| GitHub stars | 10 | 100+ |
| Risposta autori paper | — | Like/repost da 1+ autore |

---

*Versione 1.0 — Giugno 2026*
