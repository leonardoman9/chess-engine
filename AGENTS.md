# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/`: `agents/` (DQN logic), `models/` (CNN/GNN nets), `replay/` (buffers), `training/` (self-play orchestration), `utils/` (action masks, exploration, ELO math).
- Hydra configs in `conf/` (`config.yaml` entrypoint plus `experiment/`, `model/`, `training/` presets). Override via CLI or `PARAMS`.
- Experiment outputs in `results/<run>/` (checkpoints, plots, sample games) and logs in `logs/<run>/`.
- Top-level scripts: `train_hydra.py`, `analyze_training.py`, `analyze_games.py`, `evaluate_elo.py`; Docker support via `Dockerfile`, `docker-compose*.yml`; helper targets in `Makefile`.

## Build, Test, and Development Commands
- `make install` — install Python deps locally (Python 3.10+); `make build` for Docker image.
- `make train-local EXP=baseline_small [PARAMS='experiment.total_games=200']` — CPU smoke run outside Docker.
- `make train-hydra-small|medium|large` — Docker presets; `make train-hydra-gpu GPU=0 EXP=cnn_gpu_long` for GPU override.
- `make analyze-training RESULTS=results/<run>` — plots + stats; `make analyze-games` for PGN metrics.
- `make evaluate-elo-quick CHECKPOINT=.../final_model.pt` — fast Stockfish sweep; `make evaluate-elo-full` for full pass.
- `make tensorboard` / `make notebook` — developer services in-container.

## Coding Style & Naming Conventions
- Python: 4-space indent, type hints where feasible, keep functions small; prefer pure helpers in `utils/`.
- Hydra: prefer CLI overrides (`experiment=... training=...`) over copying files.
- Naming: snake_case for Python/config keys; experiments as `baseline_<size>`, `cnn_gpu_long`, `gnn_*`.
- Formatting: mirror existing style; no enforced linter—run a formatter only when touching a module holistically.

## Testing Guidelines
- Smoke test training changes with `make train-hydra-small` or `make train-local EXP=baseline_small device=cpu`.
- Validate metrics with `analyze_training.py <results_dir> --plots` and a quick ELO check via `make evaluate-elo-quick`.
- For config-only edits, ensure Hydra loads: `python train_hydra.py --cfg job --resolve`.
- Use deterministic seeds in new utilities; avoid committing large result folders.

## Commit & Pull Request Guidelines
- Commits: imperative, one topic (e.g., `tune epsilon schedule`, `fix replay masking`); mention configs touched.
- PRs: include scope, commands run, sample overrides, before/after metrics (reward curves or quick ELO), and hardware assumptions. Link issues/tasks.
- If outputs change, attach `results/<run>/experiment_info.json` and key plots; avoid committing bulky artifacts.

## Security & Configuration Tips
- Stockfish is bundled in Docker; avoid hardcoding local paths. Keep secrets out of configs and use environment variables if needed.
- Results/logs can grow quickly—clean or move old runs instead of committing them.***
