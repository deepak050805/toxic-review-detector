# Deployment Guide

This project can run locally, in Docker, or on a managed Python hosting platform. Transformer models are loaded at application startup, so choose an instance with enough memory for the configured model set.

## Local Development

```bash
git clone https://github.com/yourusername/toxic-review-detector.git
cd toxic-review-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

For macOS/Linux, activate the environment with `source venv/bin/activate`.

Open `http://localhost:5000`.

## Docker

Build and run manually:

```bash
docker build -t toxic-detector:latest .
docker run -p 5000:5000 toxic-detector:latest
```

Or use Docker Compose:

```bash
docker-compose up -d
```

## Render

1. Push the repository to GitHub.
2. Create a new Render Web Service.
3. Connect the repository.
4. Use the included `render.yaml` or configure the service manually.
5. Set production environment variables in the Render dashboard.
6. Deploy and verify `/api/health`.

## Railway

1. Push the repository to GitHub.
2. Create a Railway project from the repository.
3. Use the included `railway.json`.
4. Configure environment variables.
5. Deploy and verify `/api/health`.

## Production Checklist

- Set `FLASK_ENV=production`.
- Configure a secure `SECRET_KEY`.
- Disable debug mode.
- Use HTTPS at the platform or proxy layer.
- Add authentication before exposing moderation tools to private data.
- Add request limits for public deployments.
- Monitor memory usage during model loading and inference.
- Document data retention expectations for users.

## Operational Notes

- First deployment can take longer while dependencies and model artifacts are downloaded.
- CPU inference is supported, but GPU-backed instances can improve throughput.
- Batch requests are limited to 100 reviews to keep latency and memory usage predictable.
