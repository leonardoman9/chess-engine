# 7-day Delivery Roadmap

## Obiettivi
Preparare le evidenze sperimentali e la documentazione finale (relazione ~4-8 pagine + presentazione) da inviare alla prof.ssa Milano entro una settimana.

## Giorno 1 – Esecuzione run representative
- Lancia `make train-hydra-small` (baseline_small) su CPU per avere un run rapido (<2h) e salvare `results/<run>`.
- Avvia un run medio (`make train-hydra-medium` o `train-local EXP=baseline_medium device=cpu`) per 500 partite.
- Se disponibile, start un run esteso in Docker GPU con `make train-hydra-gpu GPU=0 EXP=cnn_gpu_long` (puoi fermarlo dopo 1000-2000 game se serve) e raccogli dati iniziali.
- Assicurati che TensorBoard scriva in `logs/` e annota le directory definitive dei risultati.

## Giorno 2 – Analisi e metriche
- Per ogni run, esegui `make analyze-training RESULTS=results/<run>` e `make evaluate-elo-quick CHECKPOINT=results/<run>/final_model.pt`.
- Salva grafici (`plots/`) e JSON (`experiment_info.json`, `training_history.json`) per la relazione.
- Registra tempi di training, epsilon finale, reward medio, variazione ELO.

## Giorno 3 – Scrivere sezione Risultati e Commenti
- Redigi paragrafi che commentano le metriche raccolte: stabilità della learning curve, gap tra CNN/GNN, impatto delle risorse (CPU vs GPU).
- Inserisci tabelle brevi con ELO stimato vs numero di game e durata run.
- Correlati i plot dalle analisi con le affermazioni (includi riferimenti a file `results/.../plots`).

## Giorno 4 – Redigere la relazione
- Trasforma `PROJECT_WORK.md` in un draft di 4–8 pagine (Markdown con paragrafi, tabelle, figure).
- Organizza la relazione così: intro/motivazioni, architettura/progettazione, setup esperimenti, risultati + commenti, conclusioni + futuri work.
- Includi passi operativi (command list) nella sezione appendice/appendice operazioni.

## Giorno 5 – Preparare presentazione
- Traduci la struttura della relazione in slide (8–12 slide) con punti chiave per ciascuna sezione e grafici evidenziati (ELO, training curves).
- Inserisci note per narrazione: motivazioni, impatto, limiti hardware.
- Salva la presentazione (PDF o formato scelto) in `presentation/` o simile.

## Giorno 6 – Revisione e rifiniture
- Rileggi relazione/presentazione, correggi stile e riferimenti.
- Controlla che README/AGENTS citino i comandi usati per riprodurre i run e aggiornali se serve.
- Verifica che i risultati siano allegabili (grafici, JSON, logs).

## Giorno 7 – Preparazione consegna
- Compila email con riepilogo e allegati (relazione, presentazione, link ai risultati).
- Riassumi i comandi usati e i principali risultati (ELO, durate).
- Prepara un breve paragrafo su prossimi possibili miglioramenti (da discutere se richiesto).
