# Post LinkedIn — RecursiveMAS Latent Visualizer
# Un post unico, completo

---

## ISTRUZIONI DI PUBBLICAZIONE

- Allega in ordine: screenshot PCA Journey, screenshot tabella statistiche
- Pubblica martedì o giovedì, ore 8-10 oppure 17-19
- Rispondi a tutti i commenti nelle prime 2 ore
- Tagga gli autori del paper se li trovi su LinkedIn: cerca "Jiaru Zou RecursiveMAS"

---

## IL POST

---

Ho visualizzato per la prima volta cosa succede davvero quando tre modelli linguistici si "parlano" nello spazio latente.

Il risultato è inaspettato.

---

**Il contesto**

RecursiveMAS è un paper di Stanford / UIUC / NVIDIA / MIT uscito ad aprile 2026.
L'idea: far collaborare più LLM senza che si scambino testo — solo vettori latenti, attraverso moduli chiamati OuterLinks.

Risultato dichiarato nel paper: +8.3% accuracy, 2.4× più veloce, −75% di token rispetto ai MAS classici.

I modelli sono open-weight su HuggingFace. Nessuno aveva ancora aperto il cofano.

---

**Cosa ho costruito**

Un visualizzatore che, durante i 3 round ricorsivi del sistema, estrae e mostra:

→ La **traiettoria PCA 2D** di tutti i vettori latenti (Planner, OuterLink12, Critic, OuterLink23, Solver) per ogni round — 15 punti in totale, aggiornati live

→ La **tabella di statistiche pipeline**: norma L2 e cosine similarity ad ogni passaggio tra agenti

[Screenshot PCA Journey]

---

**La scoperta**

Il dato più interessante è nella tabella:

| Transizione | Cosine similarity |
|---|---|
| Planner → outer_12 → Critic | **−0.015** |
| Critic → outer_23 → Solver | **−0.003** |

Le OuterLinks proiettano in modo **quasi ortogonale**.

Non portano il "significato" da un agente all'altro. Lo trasformano completamente.
Ogni agente riceve un vettore in uno spazio semantico totalmente diverso — non una versione modificata del precedente.

Questo pattern è stabile su tutti e 3 i round. Non era documentato da nessuna parte.

[Screenshot tabella statistiche]

---

**Cosa significa**

Quando il Planner "pensa" a un problema e passa il risultato al Critic, il Critic non riceve una versione compressa di quel pensiero. Riceve qualcosa di ortogonale — una proiezione in un nuovo spazio che il Critic stesso sa come elaborare.

È più simile a una traduzione tra lingue diverse che a un passaggio di un messaggio.

---

**Feature 2: Expert vs Learner**

Ho aggiunto anche un confronto tra Qwen3-8B (Expert) e Qwen3-1.7B (Learner) sullo stesso problema, con 3 round di raffinamento iterativo.

Il grafico mostra il delta di cosine similarity tra i due modelli: se scende, il modello più piccolo sta convergendo verso quello grande nello spazio latente.

---

**Link**

→ GitHub + Notebook Colab: github.com/AlexBenin01/LatentScope
→ Paper originale: arxiv.org/abs/2604.25917

Tutto open-source, modelli open-weight, zero costi per eseguirlo.

---

#MultiAgentSystems #LLM #AIResearch #OpenSource #HuggingFace #RecursiveMAS #MachineLearning

---

## VARIANTE BREVE (se vuoi qualcosa di più diretto)

Ho aperto il cofano di RecursiveMAS — il framework multi-agent di Stanford/UIUC/NVIDIA/MIT che fa collaborare LLM nello spazio latente.

Scoperta: le OuterLinks che trasferiscono informazioni tra agenti proiettano con cosine ≈ 0.
Non portano il significato. Lo trasformano completamente.

Ho costruito un visualizzatore con traiettoria PCA live e tabella di statistiche per ogni passaggio.

→ github.com/AlexBenin01/LatentScope

[Screenshot PCA Journey]
[Screenshot tabella statistiche]

#MultiAgentSystems #LLM #OpenSource #RecursiveMAS
