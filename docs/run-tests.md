# run-tests

Esegui la suite di test del progetto:

```bash
cd /path/to/recursivemas-demo
python -m pytest tests/ -v --tb=short
```

Se i test falliscono:
1. Mostra l'errore completo
2. Identifica il file e la funzione che causa il fallimento
3. Proponi una correzione minimale (< 20 righe)
4. Non modificare i test — solo il codice sorgente
