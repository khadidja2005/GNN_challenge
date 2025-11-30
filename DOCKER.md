# Docker Deployment

## Quick Start

### Production Build
```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Development Mode (with hot reload)
```bash
docker-compose --profile dev up web-dev
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `web` | 3000 | Production Next.js UI |
| `web-dev` | 3001 | Development Next.js (hot reload) |
| `gnn` | - | Python GNN environment |

## Individual Commands

### Web UI Only
```bash
cd web
docker build -t enzymes-web .
docker run -p 3000:3000 enzymes-web
```

### Python Environment Only
```bash
docker build -f Dockerfile.python -t enzymes-gnn .
docker run -it -v $(pwd)/data:/app/data enzymes-gnn bash
```

### Run GNN Scripts
```bash
# Prepare data
docker-compose run gnn python scripts/prepare_data.py

# Train baseline
docker-compose run gnn python baselines/simple_gnn.py

# Evaluate predictions
docker-compose run gnn python scripts/evaluate.py --predictions submissions/predictions.csv
```

## Volumes

- `./data` - Processed data files (train.pt, val.pt, test.pt)
- `./submissions` - Prediction files and results
- `./Dataset` - Raw ENZYMES dataset (read-only in container)

## Notes

- Production web build uses multi-stage for minimal image size
- Python container includes PyTorch and PyTorch Geometric
- Use `docker-compose down` to stop all services
