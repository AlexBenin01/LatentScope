# Post LinkedIn — RecursiveMAS Latent Visualizer

---

## POST 1 — Hook teorico
**Quando pubblicare**: oggi o domani
**Formato**: testo + screenshot della prima pagina del paper

---

Questo paper ha 5 settimane ed è già il progetto più interessante che ho visto nel 2026.

RecursiveMAS (Stanford / UIUC / NVIDIA / MIT) risolve un problema che nessun Multi-Agent System aveva ancora affrontato: il costo di far "parlare" gli agenti tra loro in testo.

La soluzione? Far collaborare gli agenti direttamente nello spazio latente, senza mai convertire i pensieri intermedi in parole.

Risultato: +8.3% accuracy, 2.4× più veloce, −75% token rispetto ai MAS classici.

E i modelli sono già su HuggingFace, open-weight e modificabili.

Ho costruito qualcosa con questo. Vi mostro cosa succede "sotto il cofano" di questi agenti.

Seguite per i prossimi aggiornamenti.

[screenshot paper]

#MultiAgentSystems #LLM #AIResearch #OpenSource #RecursiveMAS

---

**Tag suggeriti**: cerca su LinkedIn "Jiaru Zou" e gli autori del paper — metti un tag se li trovi.

---

## POST 2 — Behind the scenes
**Quando pubblicare**: 3-4 giorni dopo il Post 1
**Formato**: testo + screenshot PCA Journey (assets/screenshot_pca_journey.png) + screenshot pipeline stats (assets/screenshot_pipeline_stats.png)

---

Aggiornamento su quello che sto costruendo con RecursiveMAS.

Ho estratto i vettori latenti mentre passano da un agente all'altro durante i round ricorsivi.

[screenshot PCA Journey]

Ogni punto è la rappresentazione interna di un agente dopo quel round.
Blu = Planner, Ciano = OuterLink12, Arancio = Critic, Oro = OuterLink23, Verde = Solver.
Le frecce viola mostrano come l'output del Solver rientra nel Planner per il round successivo.

Ma la cosa più interessante è nella tabella sotto:

[screenshot pipeline stats]

Le OuterLinks — i moduli che trasferiscono informazioni tra agenti — proiettano in modo quasi ortogonale (cosine ≈ 0).

Non trasportano il "significato" da un agente all'altro. Lo trasformano completamente.

Ogni agente riceve un vettore in uno spazio semantico suo, non una versione scalata del precedente.

Questo non era mai stato visualizzato numericamente.

Il codice è open-source, link in bio.

#MultiAgentSystems #HuggingFace #OpenSource #AIResearch #RecursiveMAS

---

## POST 3 — Launch
**Quando pubblicare**: appena hai un URL HF Spaces attivo (o subito con solo GitHub + Colab)
**Formato**: testo + video screencast 60 secondi da Colab

---

Ho rilasciato la prima demo pubblica del ragionamento interno di RecursiveMAS.

[video screencast 60s]

Due feature che non esistevano prima:

**1. Monitor spazio latente**
Ogni round ricorsivo, vedi:
— La traiettoria PCA dei vettori latenti tra i 3 agenti
— La norma e la cosine similarity ad ogni passaggio (con indicatore ⊥ quando la proiezione è ortogonale)
— Entropia e confidence del Solver, aggiornati live

**2. Expert vs Learner**
Qwen3-8B vs Qwen3-1.7B sullo stesso problema, 3 round di raffinamento.
Il grafico mostra il delta di similarity: se scende, il modello più piccolo converge verso quello grande.

Tutto open-source, modelli open-weight.

→ GitHub: github.com/AlexBenin01/LatentScope
→ Notebook Colab: nel repo, cartella notebooks/

3 cose che ho scoperto costruendola:
- Le OuterLinks di RecursiveMAS proiettano quasi ortogonalmente — ogni agente opera in uno spazio semantico separato, non in una versione modificata del precedente. Non era documentato.
- Estrarre hidden states durante inference in un sistema multi-agente non è documentato quasi da nessuna parte.
- Il paper ha 5 settimane. La finestra di opportunità su ricerche fresche è reale.

#MultiAgentSystems #LLM #OpenSource #HuggingFace #AIResearch #RecursiveMAS

---

## Note operative

- **Orario ottimale**: martedì-giovedì ore 8-10 o 17-19
- **Post 1 e 2** non richiedono URL live — escono subito
- **Post 3** esce quando hai un URL (HF Spaces o anche solo Colab con ngrok temporaneo)
- Rispondi ai commenti nelle prime 2 ore dopo ogni post per massimizzare la reach
