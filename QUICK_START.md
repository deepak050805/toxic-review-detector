# Quick Start

## Local Run

```bash
git clone https://github.com/yourusername/toxic-review-detector.git
cd toxic-review-detector
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies and start Flask:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Docker Run

```bash
docker-compose up
```

Open `http://localhost:5000`.

## API Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your review text"}'
```

## Batch Example

```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["review one", "review two", "review three"]}'
```

## Common Issues

| Issue | Resolution |
|-------|------------|
| Module not found | Install dependencies in the active virtual environment |
| Port already in use | Set the `PORT` environment variable |
| Out of memory | Use a larger instance or reduce concurrency |
| Model download is slow | Wait for the initial Hugging Face download to complete |

## Related Docs

- [README.md](README.md)
- [GETTING_STARTED.md](GETTING_STARTED.md)
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
