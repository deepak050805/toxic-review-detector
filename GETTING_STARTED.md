# Getting Started

This guide covers local setup, Docker usage, and a basic API smoke test.

## Local Setup

### Windows

```bash
git clone https://github.com/yourusername/toxic-review-detector.git
cd toxic-review-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

### macOS/Linux

```bash
git clone https://github.com/yourusername/toxic-review-detector.git
cd toxic-review-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

The first run downloads the configured transformer models. Subsequent starts use the local Hugging Face cache.

## Docker

```bash
git clone https://github.com/yourusername/toxic-review-detector.git
cd toxic-review-detector
docker-compose up
```

Open `http://localhost:5000`.

## API Smoke Test

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your review text here"}'
```

Expected response shape:

```json
{
  "success": true,
  "label": "Toxic",
  "confidence": 75,
  "categories": {
    "toxicity": 75,
    "hate_speech": 60,
    "harassment": 55
  }
}
```

## Deployment

Render and Railway configuration files are included in the repository. See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) before deploying to production.

## Documentation

- [README.md](README.md) - Project overview and API reference
- [QUICK_START.md](QUICK_START.md) - Short setup reference
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment notes
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Repository layout

## Troubleshooting

| Issue | Recommended action |
|-------|--------------------|
| Dependencies fail to import | Re-run `pip install -r requirements.txt` inside the active virtual environment |
| Models fail to load | Confirm available memory and clear the Hugging Face cache if needed |
| Port 5000 is unavailable | Set a different `PORT` environment variable |
| Slow first request | Allow model download and warm-up to finish |
