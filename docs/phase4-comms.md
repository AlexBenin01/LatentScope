# Fase 4 — Comunicazione (Giorno 10)

> Carica questo file all'inizio della sessione finale.
> Obiettivo: 3 post LinkedIn pubblicati, README GitHub completo, repo pubblica.

---

## Giorno 10 — LinkedIn e documentazione

### Struttura della serie di post

#### Post 1 — Hook teorico (pubblica al Giorno 1-2)

**Formato**: testo + screenshot della pagina del paper

```
Questo paper ha 5 giorni ed è già il progetto più interessante che ho visto nel 2026.

RecursiveMAS (Stanford / UIUC / NVIDIA / MIT) risolve un problema che nessun 
Multi-Agent System aveva ancora affrontato: il costo di far "parlare" gli agenti 
tra loro in testo.

La soluzione? Far collaborare gli agenti direttamente nello spazio latente, 
senza mai convertire i pensieri intermedi in parole.

Risultato: +8.3% accuracy, 2.4× più veloce, −75% token rispetto ai MAS classici.

E i modelli sono già su HuggingFace, open-weight e modificabili.

Sto costruendo qualcosa con questo. Seguite per i prossimi aggiornamenti.

[screenshot paper]

#MultiAgentSystems #LLM #AIResearch #OpenSource
```

**Tag**: @Jiaru Zou @James Zou (cercali su LinkedIn prima di pubblicare)

---

#### Post 2 — Behind the scenes (pubblica al Giorno 5-6)

**Formato**: testo + screenshot codice + primo grafico hidden states

```
Aggiornamento sul progetto RecursiveMAS.

Ho estratto gli hidden states degli agenti durante i round ricorsivi.
Questo è quello che normalmente non si vede mai in un Multi-Agent System.

[screenshot grafico cosine similarity 3 round]

Ogni barra è un round ricorsivo. La cosine similarity misura quanto è 
cambiata la "comprensione" latente del Solver tra un round e l'altro.

Round 1 → 2: il modello cambia approccio significativamente.
Round 2 → 3: converge verso la risposta finale.

Questo succede interamente senza generare testo — solo vettori che si 
trasformano nello spazio a 1536 dimensioni del modello.

Demo quasi pronta. Prossimo aggiornamento: URL pubblico.

#MultiAgentSystems #HuggingFace #OpenSource #AIResearch
```

---

#### Post 3 — Launch (pubblica Giorno 9-10, martedì-giovedì ore 9-11 o 17-19)

**Formato**: testo + video screencast 60 secondi

```
La demo è live.

Ho costruito la prima visualizzazione pubblica del ragionamento interno 
di RecursiveMAS — il framework multi-agent ricorsivo di Stanford/UIUC/NVIDIA/MIT.

Due feature che non esistevano prima:

1. Monitor spazio latente: vedi cosine similarity, entropia e confidence 
   dei 3 agenti ad ogni round ricorsivo

2. Expert vs Learner: confronto in tempo reale tra un modello da 9B e uno 
   da 4B — il grafico mostra il delta che si riduce mentre il Learner 
   converge verso l'Expert

Tutto open-source, modelli open-weight, gratis su HuggingFace Spaces.

→ [LINK DEMO]
→ [LINK GITHUB]

3 cose che ho imparato costruendola:
- Estrarre hidden states durante inference non è documentato quasi da nessuna parte
- Il pattern Distillation del paper è quello con più potenziale applicativo
- Il paper ha 10 giorni e già nessuno aveva fatto questo — la finestra di 
  opportunità su ricerche fresche è reale

[video screencast]

#MultiAgentSystems #LLM #OpenSource #HuggingFace #AIResearch #RecursiveMAS
```

---

### README.md per GitHub

```markdown
# RecursiveMAS Latent Visualizer

Prima demo pubblica che rende visibile il ragionamento interno di un sistema 
Multi-Agent ricorsivo con RecursiveMAS.

## Demo live

→ [HuggingFace Spaces](https://huggingface.co/spaces/USERNAME/recursivemas-demo)

## Cosa fa

**Feature 1 — Monitor spazio latente**  
Visualizza cosine similarity, entropia e confidence degli hidden states del Solver 
ad ogni round ricorsivo (Sequential-Light: Qwen3-1.7B + Llama3.2-1B + Qwen2.5-Math-1.5B).

**Feature 2 — Expert vs Learner**  
Confronto side-by-side tra Expert (Qwen3.5-9B) e Learner (Qwen3.5-4B) in stile 
Distillation. Il grafico mostra il delta di similarity che converge ad ogni round.

## Setup locale

```bash
git clone https://github.com/USERNAME/recursivemas-demo
cd recursivemas-demo
pip install -r requirements.txt

# Serve token HuggingFace per scaricare i modelli RecursiveMAS
export HF_TOKEN=hf_...
python src/ui/app.py
```

## Paper originale

RecursiveMAS: Scaling Agent Collaboration through Latent-space Recursion  
Yang et al., 2026 — [arXiv:2604.25917](https://arxiv.org/abs/2604.25917)  
UIUC · Stanford University · NVIDIA · MIT

## Licenza

Apache 2.0
```

---

### Checklist finale Giorno 10

- [ ] Post 1 pubblicato (già dal giorno 1-2)
- [ ] Post 2 pubblicato (già dal giorno 5-6)
- [ ] Post 3 pubblicato con link demo e video
- [ ] README.md aggiornato su GitHub
- [ ] `STATUS.md` → `progetto: COMPLETATO`
- [ ] Risposta ai commenti nelle prime 2 ore dal Post 3

---

## Comando slash checkpoint

File `.claude/commands/checkpoint.md`:

```
Salva il progresso corrente:
1. Leggi STATUS.md e mostra lo stato attuale
2. Elenca i task completati oggi
3. Elenca i task rimanenti per la fase corrente
4. Scrivi un aggiornamento su PROGRESS.md con formato:
   ## [DATA] Sessione N
   ### Completato
   ### Da fare
   ### Blocchi/problemi
5. Non fare commit — solo aggiorna il file
```

---

*Progetto completato. Aggiorna CLAUDE.md → Fase attiva: COMPLETATA.*
