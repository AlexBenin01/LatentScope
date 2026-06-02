# STATUS.md — RecursiveMAS Latent Visualizer

Aggiorna questo file dopo ogni sessione di lavoro.

---

## Stato generale

| Campo | Valore |
|---|---|
| Fase attiva | FASE 3 — Deploy |
| Giorno corrente | 9 |
| Stato | IN CORSO |
| Ultima sessione | 2026-06-02 |
| Prossima azione | Deploy HF Spaces + conferma Feature 2 |

---

## Checklist fasi

### Fase 1 — Setup (Giorni 1–2)
- [x] G1: Colab Pro attivo, dipendenze installate, forward pass base OK
- [x] G2: Sequential-Light (Planner + Critic + Solver + 3 OuterLinks) in memoria GPU (9.7 GB)

### Fase 2 — Core (Giorni 3–7)
- [x] G3: Inference loop 3 round funzionante con OuterLink reali
- [x] G4: Hidden states estratti e verificati (similarity < 0.9999 tra round)
- [x] G5: Metriche calcolate (cosine sim, entropia, top-5 confidence)
- [x] G6: UI Gradio Feature 1 funzionante con PCA Journey live
- [x] G7: Modelli Distillation caricati (Qwen3-8B + Qwen3-1.7B)

### Fase 3 — UI e Deploy (Giorni 8–9)
- [x] G8: UI completa — Feature 1 con PCA Journey + tabella pipeline stats + barre metriche
- [x] G8: Feature 2 code completo (Expert vs Learner, streaming, grafico delta)
- [ ] G9: Deploy HF Spaces live con URL pubblico
- [ ] G9: Conferma Feature 2 end-to-end in Colab

### Fase 4 — Comunicazione (Giorno 10)
- [ ] Post 1 LinkedIn pubblicato
- [ ] Post 2 LinkedIn pubblicato
- [ ] Post 3 LinkedIn pubblicato (con link demo)
- [ ] README GitHub aggiornato con URL demo

---

## Scoperte scientifiche emerse durante lo sviluppo

- **OuterLinks proiettano quasi ortogonalmente** (cosine ≈ 0.0–0.06): le proiezioni tra agenti non preservano la direzione semantica ma la trasformano completamente. Visibile nella tabella pipeline stats con marker `⊥`.
- **Norma compressa poi espansa**: outer_12 riduce la norma del ~50-60%, il Critic la amplifica del +130%, outer_23 riduce di nuovo del ~60%, il Solver la ri-amplifica del +130-150%. Pattern stabile tra round.
- **Convergenza Planner R2→R3**: le posizioni PCA del Planner si avvicinano tra round 2 e 3, indicando stabilizzazione della rappresentazione dopo il primo round.

---

## Problemi risolti

| Problema | Soluzione |
|---|---|
| `adapter_config.json` non-standard nei modelli RecursiveMAS | `snapshot_download` a dir temporanea, `os.remove(adapter_config.json)` prima del load |
| OuterLink architettura errata (1 layer → 5 layer effettivi) | Inferita da `proj1.weight.shape[0]` (mid_dim), `ln_target` normalizza `out_dim` non `mid_dim` |
| Solver genera output fuori tema | Risposta generata dal Planner con stato latente Solver proiettato via outer_31 |
| Qwen3-4B architettura ibrida Mamba incompatibile | Sostituito con Qwen3-1.7B (stessa famiglia del Planner) |
| `qwen3_5_text` non supportato da transformers | Sostituiti Qwen3.5-9B/4B con Qwen3-8B/1.7B (architetture standard) |
| DEVICE statico incompatibile con ZeroGPU | Rimpiazzato con `_dev()` valutato a runtime; OuterLinks su CPU al load, spostati a GPU all'inference |

---

## Note di sessione

*Sessione 2026-06-02: Feature 1 completa e visivamente impressionante. OuterLinks ortogonali confermati dai dati. Da fare: conferma Feature 2 in Colab + deploy HF Spaces.*
