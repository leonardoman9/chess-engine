# 🐳 Docker Setup per Chess-RL Training

Questo documento spiega come utilizzare Docker per il training del chess engine, ottimizzato per server con GPU A40.

## 🚀 Quick Start

### 1. Build del progetto
```bash
make build
```

### 2. Test Phase 1 (verifica implementazione DQN)
```bash
make test-phase1
```

### 3. Sessione interattiva per development
```bash
make interactive
```

### 4. Monitoring con TensorBoard
```bash
make tensorboard
# Apri http://localhost:6006
```

## 🖥️ Server Deployment (A40 GPU)

### Prerequisiti Server
1. **NVIDIA Docker runtime** installato
2. **Docker Compose v2.0+** con support per `runtime: nvidia`
3. **NVIDIA drivers** compatibili con CUDA 12.1
4. **48GB VRAM** disponibili su A40

### Setup Iniziale Server
```bash
# Clone repository
git clone <repo-url>
cd chess-engine

# Build e test iniziale
make server-setup

# Verifica GPU access
make server-test
```

### Comandi Server
```bash
# Test con monitoring GPU
make server-test

# Sessione interattiva con GPU info
make server-interactive

# Monitor GPU usage durante training
make gpu-monitor
```

## 📋 Struttura Docker Services

### `chess-dqn-training`
- **Uso:** Training automatico e testing
- **GPU:** Accesso completo A40
- **Volumes:** Modelli, logs, checkpoints
- **Comando default:** `python test_dqn_phase1.py`

### `chess-dqn-interactive`
- **Uso:** Development e debugging
- **GPU:** Accesso completo A40
- **TTY:** Sessione bash interattiva
- **Ports:** TensorBoard (6006), Jupyter (8888)

### `tensorboard`
- **Uso:** Monitoring training metrics
- **Accesso:** http://localhost:6006
- **Logs:** `/app/logs` volume

### `backend` + `frontend`
- **Uso:** API e web interface (legacy)
- **GPU:** Non necessaria
- **Ports:** API (8000), Frontend (3000)

## 🗂️ Volume Mapping

```
./models       → /app/models       # Modelli salvati
./checkpoints  → /app/checkpoints  # Training checkpoints
./logs         → /app/logs         # TensorBoard logs
./results      → /app/results      # Evaluation results
./data         → /app/data         # Training data
```

## ⚙️ Environment Variables

### GPU Configuration
- `NVIDIA_VISIBLE_DEVICES=all` - Accesso a tutte le GPU
- `CUDA_VISIBLE_DEVICES=0` - Usa solo GPU 0
- `TORCH_BACKENDS_CUDNN_BENCHMARK=true` - Ottimizzazione cuDNN

### Python Configuration
- `PYTHONPATH=/app` - Python module path
- Poetry configurato per non creare venv

## 🧪 Testing Pipeline

### Phase 1 Testing
```bash
# Build e test completo
make build
make test-phase1

# Output atteso:
# ✅ Dueling DQN tests passed!
# ✅ Action space tests passed!
# ✅ Exploration tests passed!
# ✅ Replay buffer tests passed!
# ✅ DQN Agent tests passed!
# ✅ Integration tests passed!
# 🎉 ALL PHASE 1 TESTS PASSED!
```

### GPU Verification
```bash
make server-test

# Output atteso:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.1   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |   0  NVIDIA A40           On   | 00000000:xx:xx.x Off |                    0 |
# | xxx   xxC    P8    xxW / xxxW |      0MiB / 46068MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

## 🔧 Development Workflow

### 1. Sviluppo Locale
```bash
# Build dell'immagine
make build

# Test componenti
make test-phase1

# Development interattivo
make interactive
```

### 2. Sul Server A40
```bash
# Setup iniziale
make server-setup

# Development con GPU
make server-interactive
```

### 3. Training Pipeline
```bash
# Avvia training (quando pronto)
make train-dqn

# Monitor con TensorBoard
make tensorboard

# Monitor GPU usage
make gpu-monitor
```

## 📊 Monitoring e Debugging

### TensorBoard
- URL: http://localhost:6006
- Logs: `./logs/` directory
- Metrics: Loss, ELO, epsilon, Q-values

### Container Logs
```bash
# Logs training container
docker-compose logs chess-dqn-training

# Follow logs real-time
docker-compose logs -f chess-dqn-training
```

### GPU Monitoring
```bash
# Inside container
nvidia-smi

# From host
make gpu-monitor
```

## 🚨 Troubleshooting

### GPU Non Riconosciuta
```bash
# Verifica NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Verifica docker-compose
docker-compose config
```

### Memory Issues
- Reduce `batch_size` in DQN config
- Monitor with `nvidia-smi`
- Check container memory limits

### Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./models ./checkpoints ./logs ./results
```

## 📦 Production Deployment

### Server Requirements
- **GPU:** NVIDIA A40 (48GB VRAM)
- **RAM:** 64GB+ recommended
- **Storage:** 500GB+ SSD
- **Docker:** 20.10+ with nvidia runtime
- **CUDA:** 12.1 compatible drivers

### Deployment Checklist
- [ ] NVIDIA Docker runtime configured
- [ ] GPU access verified (`nvidia-smi`)
- [ ] Repository cloned
- [ ] Docker image built successfully
- [ ] Phase 1 tests pass
- [ ] TensorBoard accessible
- [ ] Volume permissions correct
- [ ] Monitoring setup ready

### Resource Monitoring
```bash
# Memory usage
docker stats

# GPU utilization
make gpu-monitor

# Disk space
df -h
du -sh ./checkpoints ./logs
```

## 🔄 Updating Code

Sul server:
```bash
# Pull latest changes
git pull

# Rebuild image
make build

# Test changes
make test-phase1
```

## 📝 Note Importanti

1. **GPU Memory:** A40 ha 48GB, usare batch size fino a 512
2. **Training Data:** Stored in volumes, persiste tra restart
3. **Checkpoints:** Auto-salvati durante training
4. **TensorBoard:** Logs real-time, accessibile via web
5. **Interactive Mode:** Utile per debugging e development

Per domande specifiche, vedere `tasklist.md` o consultare i log del container.
