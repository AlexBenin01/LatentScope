# Fase 2 — Sviluppo Core (Giorni 3–7)

> Carica questo file all'inizio di ogni sessione in questa fase.
> Obiettivo: inference ricorsiva funzionante + estrazione hidden states + metriche.

---

## Giorno 3 — Inference loop base

### Cosa implementare

Il loop ricorsivo a 3 round: ogni round i 3 agenti si passano hidden states tramite OuterLink.
Solo il round 3 decodifica in testo. I round 1 e 2 passano tensori grezzi.

### File da creare: `src/inference/sequential_loop.py`

```python
import torch

def run_recursive_loop(question: str, models: dict, n_rounds: int = 3) -> dict:
    """
    Esegue il loop ricorsivo Sequential-Light.
    
    Args:
        question: domanda in testo
        models: dict da load_sequential_light()
        n_rounds: numero di round ricorsivi (default 3)
    
    Returns:
        dict con:
          - 'answer': risposta testuale finale (solo round n_rounds)
          - 'hidden_states': list[tensor] — uno per round (hidden state Solver)
          - 'logits': list[tensor] — logits finali per round
    """
    results = {
        "answer": None,
        "hidden_states": [],
        "logits": []
    }
    
    # Tokenizza input iniziale
    inputs = models["planner_tokenizer"](
        question, return_tensors="pt"
    ).to("cuda")
    
    # Stato latente condiviso tra round (inizializzato a None)
    latent_state = None
    
    for round_idx in range(1, n_rounds + 1):
        is_last_round = (round_idx == n_rounds)
        
        # --- PLANNER ---
        with torch.no_grad():
            planner_out = models["planner"](
                **inputs,
                output_hidden_states=True
            )
        planner_hidden = planner_out.hidden_states[-1]  # ultimo layer
        
        # --- OUTERLINK: Planner -> Critic ---
        # TODO: sostituire con la chiamata reale all'OuterLink della repo
        critic_input_hidden = planner_hidden  # placeholder
        
        # --- CRITIC ---
        with torch.no_grad():
            critic_out = models["critic"](
                inputs_embeds=critic_input_hidden,
                output_hidden_states=True
            )
        critic_hidden = critic_out.hidden_states[-1]
        
        # --- OUTERLINK: Critic -> Solver ---
        solver_input_hidden = critic_hidden  # placeholder
        
        # --- SOLVER ---
        with torch.no_grad():
            solver_out = models["solver"](
                inputs_embeds=solver_input_hidden,
                output_hidden_states=True
            )
        
        # Salva hidden state del Solver per questo round
        results["hidden_states"].append(solver_out.hidden_states[-1].detach().cpu())
        results["logits"].append(solver_out.logits.detach().cpu())
        
        # Solo l'ultimo round decodifica in testo
        if is_last_round:
            output_ids = solver_out.logits.argmax(dim=-1)
            results["answer"] = models["solver_tokenizer"].decode(
                output_ids[0], skip_special_tokens=True
            )
    
    return results
```

> **TODO critico al giorno 3**: leggere il codice ufficiale di `RecursiveMAS/RecursiveMAS` su GitHub
> per capire come caricare e chiamare correttamente l'OuterLink. I `# placeholder` vanno sostituiti.

### Checkpoint Giorno 3

`run_recursive_loop("Quanto fa 2+2?", models)` restituisce un dict con:
- `answer` non vuoto
- `hidden_states` lista di 3 tensori
- `logits` lista di 3 tensori

---

## Giorno 4 — Estrazione hidden states (passaggio critico)

> Questo è il giorno più delicato. Prenditi tempo e verifica ogni shape.

### File da creare: `src/inference/hidden_states.py`

```python
import torch
from typing import List

def extract_hidden_states(run_results: dict) -> List[torch.Tensor]:
    """
    Estrae e normalizza gli hidden states del Solver per round.
    
    Shape attesa per Qwen2.5-Math-1.5B: [batch, seq_len, 1536]
    Ritorna lista di tensori con shape [1536] (media sul seq_len)
    """
    normalized = []
    
    for round_idx, hs in enumerate(run_results["hidden_states"]):
        # hs shape: [batch_size, seq_len, hidden_dim]
        # Prendi l'ultimo token (posizione -1) come rappresentazione del round
        last_token_hs = hs[:, -1, :]  # shape: [batch, hidden_dim]
        
        # Squeeze batch dimension
        vec = last_token_hs.squeeze(0)  # shape: [hidden_dim]
        
        normalized.append(vec)
        print(f"Round {round_idx+1} hidden state shape: {vec.shape}")
    
    return normalized


def verify_hidden_states_differ(hidden_states: List[torch.Tensor]) -> bool:
    """
    Sanity check: gli hidden states dei round devono essere diversi.
    Se sono identici, c'è un bug nel loop ricorsivo.
    """
    if len(hidden_states) < 2:
        return True
    
    for i in range(1, len(hidden_states)):
        sim = torch.nn.functional.cosine_similarity(
            hidden_states[i-1].unsqueeze(0),
            hidden_states[i].unsqueeze(0)
        ).item()
        print(f"Similarity round {i} vs {i+1}: {sim:.4f}")
        
        if sim > 0.9999:
            print("⚠️  WARNING: hidden states quasi identici — bug nel loop?")
            return False
    
    print("✓ Hidden states verificati: cambiano tra i round")
    return True
```

### Checkpoint Giorno 4

`verify_hidden_states_differ(hidden_states)` stampa `✓` senza warning.
Le 3 similarity tra round devono essere < 0.9999 e idealmente < 0.98.

---

## Giorno 5 — Calcolo metriche

### File da creare: `src/inference/metrics.py`

```python
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
import numpy as np
from typing import List, Dict

def compute_round_metrics(
    hidden_states: List[torch.Tensor],
    logits: List[torch.Tensor]
) -> List[Dict[str, float]]:
    """
    Calcola le metriche per ogni round ricorsivo.
    
    Returns:
        Lista di dict, uno per round, con:
          - cosine_sim: similarity con il round precedente (round 1 = None)
          - entropy: entropia della distribuzione softmax sull'ultimo token
          - confidence: probabilità del token più probabile
    """
    metrics = []
    
    for i, (hs, lgt) in enumerate(zip(hidden_states, logits)):
        m = {}
        
        # Cosine similarity con round precedente
        if i == 0:
            m["cosine_sim"] = None  # nessun confronto per il round 1
        else:
            m["cosine_sim"] = F.cosine_similarity(
                hidden_states[i-1].unsqueeze(0),
                hs.unsqueeze(0)
            ).item()
        
        # Entropia e confidence sull'ultimo token
        last_token_logits = lgt[0, -1, :]  # shape: [vocab_size]
        probs = F.softmax(last_token_logits, dim=-1).numpy()
        
        m["entropy"] = float(scipy_entropy(probs))
        m["confidence"] = float(probs.max())
        
        metrics.append(m)
        print(f"Round {i+1}: sim={m['cosine_sim']}, entropy={m['entropy']:.3f}, conf={m['confidence']:.3f}")
    
    return metrics


def compute_expert_learner_delta(
    expert_hidden: List[torch.Tensor],
    learner_hidden: List[torch.Tensor]
) -> List[float]:
    """
    Calcola il delta di similarity Expert vs Learner per round.
    Il delta dovrebbe ridursi ad ogni round (convergenza).
    """
    deltas = []
    for i, (e, l) in enumerate(zip(expert_hidden, learner_hidden)):
        # Proietta allo stesso spazio se hidden dim diversa
        min_dim = min(e.shape[-1], l.shape[-1])
        sim = F.cosine_similarity(
            e[:min_dim].unsqueeze(0),
            l[:min_dim].unsqueeze(0)
        ).item()
        delta = 1.0 - sim  # 0 = identici, 1 = opposti
        deltas.append(delta)
        print(f"Round {i+1} Expert-Learner delta: {delta:.4f}")
    
    return deltas
```

### Checkpoint Giorno 5

`compute_round_metrics(hidden_states, logits)` stampa 3 righe senza errori.
Trend atteso: entropy decresce, confidence cresce da round 1 a round 3.

---

## Giorno 7 — Modelli Distillation

### File da creare: `src/models/load_distillation.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_distillation(device="cuda"):
    """
    Carica Expert (9B) e Learner (4B) con Outerlinks.
    ATTENZIONE: richiede ~26 GB VRAM — usare bfloat16 obbligatorio.
    """
    models = {}
    
    for role, model_id in [
        ("learner", "RecursiveMAS/Distillation-Learner-Qwen3.5-4B"),
        ("expert",  "RecursiveMAS/Distillation-Expert-Qwen3.5-9B"),
    ]:
        print(f"Caricamento {role}...")
        models[f"{role}_tokenizer"] = AutoTokenizer.from_pretrained(model_id)
        models[role] = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True
        )
        models[role].eval()
        
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM dopo {role}: {used:.1f} GB")
    
    return models
```

> **Se OOM con entrambi in memoria**: caricare Expert, fare inference, salvare hidden states su disco,
> scaricare Expert, caricare Learner, fare inference. Più lento ma funziona su 40GB.

### Checkpoint Giorno 7

`compute_expert_learner_delta(expert_hs, learner_hs)` stampa 3 delta.
Il delta al round 3 deve essere **minore** del delta al round 1.

---

## Test obbligatori per questa fase

File: `tests/test_shapes.py`

```python
import torch
import pytest

def test_hidden_state_shape():
    """Hidden state Solver deve avere shape [hidden_dim] dopo normalization."""
    dummy = torch.randn(1, 10, 1536)  # [batch, seq, hidden]
    last = dummy[:, -1, :].squeeze(0)
    assert last.shape == (1536,), f"Shape errata: {last.shape}"

def test_metrics_keys():
    """compute_round_metrics deve restituire dict con chiavi corrette."""
    from src.inference.metrics import compute_round_metrics
    hs = [torch.randn(1536) for _ in range(3)]
    lg = [torch.randn(1, 5, 32000) for _ in range(3)]
    metrics = compute_round_metrics(hs, lg)
    assert len(metrics) == 3
    assert all("entropy" in m and "confidence" in m for m in metrics)

def test_delta_decreases():
    """Il delta Expert-Learner deve essere monitorabile (non NaN)."""
    from src.inference.metrics import compute_expert_learner_delta
    e_hs = [torch.randn(4096) for _ in range(3)]
    l_hs = [torch.randn(3072) for _ in range(3)]
    deltas = compute_expert_learner_delta(e_hs, l_hs)
    assert len(deltas) == 3
    assert all(0 <= d <= 2 for d in deltas)
```

Esegui con: `/project:run-tests`

---

*Prossima fase: `.claude/phases/phase3-ui.md`*
