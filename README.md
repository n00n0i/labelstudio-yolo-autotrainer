# Label Studio + YOLO Auto-Training

Automatically train YOLO models when annotations reach a threshold in Label Studio.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/labelstudio-yolo.git
cd labelstudio-yolo
./setup.sh
```

## How It Works

```
┌──────────────┐     Webhook      ┌──────────────┐
│ Label Studio │ ───────────────▶ │ Auto-Trainer │
│  (Annotate)  │                  │  (Trigger)   │
└──────────────┘                  └──────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │   YOLO Training      │
                              │   (GPU Worker)       │
                              └──────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │   Model Registry     │
                              │   (Auto-deploy)      │
                              └──────────────────────┘
```

## Workflow

1. **Annotate** - Label images in Label Studio
2. **Trigger** - Webhook fires at threshold (default: 100 annotations)
3. **Export** - Auto-export in YOLO format
4. **Train** - YOLO training starts on GPU worker
5. **Deploy** - Model saved to registry (auto-deploy optional)

## Configuration

### Environment Variables

| Variable | Default | Description |
|:---|:---:|:---|
| `TRAINING_THRESHOLD` | 100 | Min annotations to trigger training |
| `LABEL_STUDIO_TOKEN` | - | API token for Label Studio |
| `DATASET_DIR` | ./datasets | Dataset storage |
| `MODELS_DIR` | ./models | Model output directory |

### Training Config

Edit `webhook_handler.py`:

```python
config = TrainingConfig(
    model_type="yolov8",    # yolov8, yolov9
    model_size="n",         # n, s, m, l, x
    epochs=100,
    batch_size=16,
    auto_deploy=False       # Auto-deploy to inference
)
```

## API Endpoints

### Webhook
```bash
POST /webhook/labelstudio
# Label Studio sends annotation events here
```

### Training Status
```bash
GET /training/status              # All trainings
GET /training/status/{project_id} # Specific project
```

### Manual Trigger
```bash
POST /training/trigger/{project_id}
```

## Monitoring

```bash
# Check training status
curl http://localhost:8000/training/status

# View worker logs
docker-compose logs -f yolo-worker

# View webhook logs
docker-compose logs -f auto-trainer
```

## Project Structure

```
labelstudio-yolo/
├── docker-compose.yml          # Services
├── webhook_handler.py          # FastAPI webhook handler
├── yolo-worker/                # GPU training worker
│   ├── Dockerfile
│   └── scripts/worker.py
├── setup.sh                    # Quick setup
└── README.md                   # This file
```

## Requirements

- Docker & Docker Compose
- NVIDIA GPU (for training)
- NVIDIA Container Toolkit

## Troubleshooting

### Training not triggering
- Check webhook URL in Label Studio settings
- Verify `TRAINING_THRESHOLD` reached
- Check logs: `docker-compose logs auto-trainer`

### GPU not detected
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Low accuracy
- Increase `TRAINING_THRESHOLD` (need more data)
- Adjust `epochs`, `batch_size`
- Try larger model size (`s`, `m`, `l`)

## License

MIT - For research use only
