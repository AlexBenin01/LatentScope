# Fase 1 — Setup (Giorni 1–2)

> Carica questo file all'inizio di ogni sessione in questa fase.
> Obiettivo: ambiente operativo pronto, modelli Sequential-Light in memoria GPU.

---

## Giorno 1 — Setup ambiente

### Task da completare (in ordine)

- [ ] Creare notebook `notebooks/day1_setup.ipynb` su Colab Pro
- [ ] Selezionare runtime A100 40GB (Runtime → Cambia tipo di runtime → A100)
- [ ] Installare dipendenze

```python
!pip install transformers torch gradio scipy matplotlib huggingface_hub accelerate
```

- [ ] Autenticarsi su HuggingFace

```python
from huggingface_hub import login
login(token="hf_...")  # token con permessi read
```

- [ ] Eseguire forward pass di verifica su modello base

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

inputs = tokenizer("Ciao, come stai?", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)

print("Shape logits:", outputs.logits.shape)  # atteso: [1, seq_len, vocab_size]
print("Giorno 1 OK ✓")
```

### Checkpoint Giorno 1

Il task è completato quando il print finale mostra `Giorno 1 OK ✓` senza errori CUDA.

---

## Giorno 2 — Caricamento modelli Sequential-Light

### Modelli da scaricare (in ordine per VRAM)

| Ordine | Modello HuggingFace | Ruolo | VRAM stimata |
|---|---|---|---|
| 1 | `RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B` | Solver | ~3 GB |
| 2 | `RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B` | Critic | ~2 GB |
| 3 | `RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B` | Planner (base model) | ~3.5 GB |
| 4 | `RecursiveMAS/Sequential-Light-Outerlinks` | RecursiveLink esterno | ~200 MB |

> **Nota**: caricare prima i modelli più piccoli per verificare che la VRAM regga prima di caricare tutto.

### Template caricamento (da mettere in `src/models/load_sequential.py`)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sequential_light(device="cuda"):
    """
    Carica i 3 modelli Sequential-Light + Outerlinks.
    Restituisce un dict con modelli e tokenizer pronti.
    """
    models = {}
    
    configs = [
        ("solver",  "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B"),
        ("critic",  "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B"),
        ("planner", "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B"),  # o modello base Qwen3-1.7B
    ]
    
    for role, model_id in configs:
        print(f"Caricamento {role}: {model_id}...")
        models[f"{role}_tokenizer"] = AutoTokenizer.from_pretrained(model_id)
        models[role] = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True  # CRITICO: abilita estrazione hidden states
        )
        models[role].eval()
    
    # Verifica VRAM
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM usata: {used:.1f} GB / {total:.0f} GB disponibili")
    
    return models
```

> **Se OOM**: usare `load_in_8bit=True` come fallback — aggiunge `bitsandbytes` alle dipendenze.

### Checkpoint Giorno 2

Tutti e 4 i modelli caricati senza OOM. Stampa VRAM usata < 12 GB.
Aggiorna `STATUS.md` → `fase1: COMPLETATA`.

---

## Comandi slash disponibili in questa fase

- `/project:checkpoint` — salva stato attuale su `PROGRESS.md`
- `/project:run-tests` — esegue `tests/test_shapes.py`

---

*Prossima fase: `.claude/phases/phase2-core.md`*
