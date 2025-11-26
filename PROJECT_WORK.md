# Project Work Report & Roadmap

## Motivazioni del progetto
Rivolgere una pipeline di reinforcement learning al dominio scacchistico permette di unire le nozioni di DQN/GNN viste nel corso con una problematica reale: esplorare come agenti self-play con replay, target network e masking si comportano contro Stockfish, pur su risorse studentesche. Il lavoro punta anche a fare esperienza con Hydra, Docker e strumenti di analisi (TensorBoard, ELO, analisi partite).

## Argomenti trattati
- Deep Q-Network dueling con backbone CNN e variante GNN (Torch Geometric).
- Hydra per gestire esperimenti/config override (`conf/experiment`, `conf/model`, `conf/training`).
- Self-play con epsilon-greedy, reward shaping, Stockfish come avversario di eval.
- Analisi post-run: `analyze_training.py`, `analyze_games.py`, `evaluate_elo.py`, TensorBoard logs in `logs/`.

## Progettazione
La pipeline eccita una versione locale (CPU) e Dockerizzata (GPU). Il Makefile organizza i target (`train-local`, `train-hydra-*`, `train-hydra-gpu`, `analyze-training`, `evaluate-elo-*`, `tensorboard`). Hydra costruisce configurazioni con `experiment`, `model`, `training`, e applica override CLI.

## Architettura
Il codice principale sta in `src/`: `agents/` (logiche agent DQN), `models/` (CNN/GNN), `replay/`, `training/` (self-play, configurazioni), `utils/` (action mapping, esplorazione, ELO). `train_hydra.py` coordina self-play, checkpoint e logging; `conf/` descrive i profili; `results/<run>/` raccoglie output.

## Realizzazione
- `train_hydra.py` legge Hydra, genera partite con epsilon-annealing e target network. Salva checkpoints, buffer e metriche.
- Docker Compose (`docker-compose.yml` e `docker-compose.server.yml`) incapsula l’ambiente con PyTorch, Stockfish, Torch Geometric (per GNN) e TensorBoard.
- Analisi con `analyze_training.py` (risultati/plots), `analyze_games.py` (metriche gioco) e `evaluate_elo.py` (sweep Stockfish); sample games in `results/<run>/sample_games`.

## Tecnologie realizzate
- Python 3.10+, PyTorch, Torch Geometric, python-chess, Stockfish 11+, Hydra.
- Containerization via Docker + Docker Compose; helper Makefile per build/run/servizi.
- Logging/monitoring: TensorBoard (`logs/`), Hydra + JSON metadata (`experiment_info.json`, `training_history.json`), ELO evaluation combinata.

## Risultati
La pipeline gira localmente su CPU (100–2000 partite) e su GPU (rampanti 10k). La CNN/ GNN completano self-play con replay e reward shaping; si producono checkpoint (`results/<run>/final_model.pt`), buffer, log e Grafici ELO. Le metriche possono essere riviste con `make analyze-training RESULTS=...` e `make evaluate-elo-quick CHECKPOINT=...`.

## Conclusioni
Il project work implementa l’intero ciclo di training/valutazione, con Hydra configurabile e supporto Docker/TensorBoard. Sebbene i limiti hardware impediscano di raggiungere l’ELO target (1100), la pipeline è pronta per scale-up su infrastrutture più potenti.

## Futuri miglioramenti possibili
1. Migliorare l’accuratezza delle GNN (Torch Geometric) con nuovi tipi di pooling/attention e esperimenti multi-gpu.
2. Integrare logging avanzato (TensorBoard custom, profiling) e sistemi di resumable checkpoint per training prolungati.
3. Automatizzare l’analisi dei risultati e la generazione di relazione (es. script che raccoglie metriche chiave e screenshot).
4. Stabilire test automatici per moduli critici (`src/utils/*`, replay buffer) e validare configurazioni Hydra con `--cfg job`.

## Roadmap delle consegne
1. **Relazione tecnica** – Tradurre questa struttura in un documento formale con metriche concrete (numero di partite, durata run, ELO).
2. **Presentazione** – Preparare slide che coprano motivazioni, architettura, risultati (grafici da `results/<run>/plots`), limiti, next steps.
3. **Codice e istruzioni** – Verificare che README/AGENTS contengano i passi operativi (es. `make build`, `python train_hydra.py experiment=baseline_small device=cpu`, `make evaluate-elo-quick CHECKPOINT=...`). Allegare log/screenshot (TensorBoard) e `results/<run>/experiment_info.json`.
4. **Feedback finale** – Chiedere alla professoressa eventuali chiarimenti, fornendo documenti e tracce log, e pianificare eventuali migliorie successive.
