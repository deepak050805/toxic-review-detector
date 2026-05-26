# Toxic Review Detector

Toxic Review Detector is a Flask-based moderation dashboard for analyzing user reviews with pretrained transformer encoders. It exposes a web interface, REST API endpoints, category-level moderation scores, and deployment-ready configuration for containerized Python hosting.

## Project Overview

The application classifies review text for toxic content, hate speech indicators, and harassment-related language. It is structured as a compact SaaS-style moderation platform: the frontend collects review text, the Flask API validates and preprocesses it, and the model layer runs transformer inference through Hugging Face pipelines.

This repository is intended to demonstrate a deployable AI/NLP application rather than a notebook-only model experiment. The code separates routing, preprocessing, model loading, frontend behavior, and deployment configuration so the project remains easy to extend.

## Features

- Transformer-based toxicity detection
- Category scores for toxicity, hate speech, and harassment signals
- Interactive dashboard with loading, error, and result states
- Chart.js visualizations for moderation score distribution
- Single-review and batch prediction endpoints
- Local inference without third-party moderation API calls
- Text validation and preprocessing before inference
- Docker, Render, Railway, and process-based deployment support
- Responsive dark dashboard interface

## Architecture

```text
Frontend Dashboard
        |
        v
Flask REST API
        |
        v
Text Preprocessing Layer
        |
        v
Transformer Inference Layer
        |
        v
Toxicity Classification Engine
        |
        v
JSON Response Pipeline
```

### Request Lifecycle

1. The user submits review text from the dashboard.
2. `static/js/script.js` validates length constraints and sends a JSON request to `/api/predict`.
3. `app.py` validates the payload and passes text through `utils/text_processing.py`.
4. `models/model_loader.py` runs pretrained transformer pipelines through Hugging Face Transformers and PyTorch.
5. The Flask API normalizes model scores into a consistent JSON response.
6. The frontend updates result cards, progress indicators, charts, and report export state.

### Repository Layout

```text
Toxic Review Detector/
|-- app.py                       Flask routes, API response flow, inference orchestration
|-- config.py                    Environment-specific Flask configuration
|-- requirements.txt             Runtime dependency pins
|-- models/
|   |-- model_loader.py          Hugging Face pipeline loading and model scoring
|-- utils/
|   |-- text_processing.py       Text cleaning, validation, and keyword helpers
|-- templates/
|   |-- index.html               Dashboard markup rendered by Flask
|-- static/
|   |-- css/style.css            Responsive dashboard theme and component styles
|   |-- js/script.js             Fetch workflow, state management, charts, animations
|-- Dockerfile                   Production container image definition
|-- docker-compose.yml           Local container orchestration
|-- render.yaml                  Render Web Service configuration
|-- railway.json                 Railway deployment configuration
|-- Procfile                     Process command for compatible Python hosts
```

## Tech Stack

**Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js

**Backend:** Python, Flask, Gunicorn

**Machine Learning:** PyTorch, Hugging Face Transformers

**Deployment:** Docker, Docker Compose, Render, Railway, GitHub Actions

## Model Layer

Encoder transformer models are suitable for this application because moderation classification depends on contextual language understanding rather than keyword matching alone. BERT, DistilBERT, and RoBERTa-style encoders produce contextual representations that can capture phrasing, sentiment, and abusive language patterns in short-form review text.

Configured pipelines:

- `unitary/toxic-bert` for toxicity, identity-hate, insult, and obscene-language signals
- `lxyuan/distilbert-base-multilingual-cased-sentiments-student` for sentiment classification

The model layer returns normalized scores on a `0-100` scale. The API uses the toxicity score as the primary confidence value and averages hate speech and abusive-language signals for the harassment category. These values are confidence-style model outputs, not benchmarked accuracy metrics.

## Installation

### Prerequisites

- Python 3.8+
- Git
- 4 GB RAM minimum, 8 GB recommended for smoother model loading
- Approximately 2 GB disk space for downloaded model artifacts

### Local Setup

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

Install dependencies and run the application:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in a browser. The first run downloads the configured transformer models and can take several minutes depending on network speed.

## Usage

### Web Dashboard

Use the dashboard to submit review text, inspect the overall classification, compare category scores, view score distribution charts, and download a plain-text moderation report.

### API Flow

The API accepts JSON requests and returns structured moderation results. Invalid payloads return a `400` response. Model loading failures return a `500` response with a stable error shape.

#### Health Check

```http
GET /api/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-05-26T10:30:45.123456"
}
```

#### Single Prediction

```http
POST /api/predict
Content-Type: application/json
```

Request:

```json
{
  "text": "Your review text here"
}
```

Response:

```json
{
  "success": true,
  "label": "Toxic",
  "confidence": 75,
  "categories": {
    "toxicity": 75,
    "hate_speech": 60,
    "harassment": 55
  },
  "timestamp": "2026-05-26T10:30:45.123456"
}
```

#### Batch Prediction

```http
POST /api/batch
Content-Type: application/json
```

Request:

```json
{
  "texts": ["review one", "review two", "review three"]
}
```

Response:

```json
{
  "success": true,
  "count": 3,
  "results": [
    {
      "text": "review one",
      "label": "Clean",
      "confidence": 12,
      "categories": {
        "toxicity": 12,
        "hate_speech": 4,
        "harassment": 8
      }
    }
  ]
}
```

Maximum batch size: 100 reviews.

#### Error Response

```json
{
  "success": false,
  "error": "Text must be at least 5 characters"
}
```

## File Documentation

### `app.py`

**Purpose:** Defines the Flask application, serves the dashboard, exposes REST endpoints, and coordinates preprocessing with model inference.

**Architecture role:** Acts as the API boundary between the browser dashboard and the NLP inference layer. It imports `TextProcessor` for validation/cleaning and `get_model()` for transformer predictions.

**Core functionality:** Handles `/`, `/api/health`, `/api/predict`, and `/api/batch`; normalizes prediction scores; returns consistent JSON success and error responses.

**Maintainability notes:** The shared `analyze_text()` helper keeps single and batch inference consistent. Future production versions can add authentication, request tracing, rate limiting, and asynchronous queues without changing the frontend contract.

### `config.py`

**Purpose:** Centralizes environment-specific Flask settings.

**Architecture role:** Provides configuration classes for development, production, and testing. It keeps runtime settings separate from route logic.

**Core functionality:** Loads `.env` values through `python-dotenv`, defines default flags, and exposes `get_config()` for environment selection.

**Maintainability notes:** Additional deployment-specific settings, such as CORS, rate limits, or model identifiers, can be added here as the application grows.

### `models/model_loader.py`

**Purpose:** Loads and manages the Hugging Face text-classification pipelines.

**Architecture role:** Owns the transformer inference layer. The Flask API calls this module instead of interacting directly with PyTorch or model objects.

**Core functionality:** Selects CPU or CUDA, initializes toxicity/hate-speech/abusive-language pipelines, converts model scores to percentages, and exposes a singleton loader through `get_model()`.

**Maintainability notes:** Model names are currently fixed in code for clarity. For larger deployments, move model selection into environment configuration and add warm-up/health diagnostics.

### `utils/text_processing.py`

**Purpose:** Provides deterministic text cleaning and validation before inference.

**Architecture role:** Sits between the API request layer and the model layer, ensuring the model receives bounded, normalized text.

**Core functionality:** Removes URLs and email addresses, normalizes whitespace, validates length constraints, and includes lightweight keyword extraction support.

**Maintainability notes:** This module is intentionally framework-independent, making it easy to test or reuse in background jobs.

### `templates/index.html`

**Purpose:** Defines the server-rendered dashboard structure.

**Architecture role:** Flask renders this file for the main route. It references static CSS and JavaScript assets and provides the DOM elements updated by the frontend script.

**Core functionality:** Includes the hero, review input form, loading/error states, result cards, category cards, chart canvases, feature summaries, processing flow, and API surface section.

**Maintainability notes:** IDs used by `static/js/script.js` are part of the frontend contract. Rename them carefully and update JavaScript selectors at the same time.

### `static/js/script.js`

**Purpose:** Controls client-side dashboard behavior.

**Architecture role:** Bridges the browser UI and Flask API through the Fetch API.

**Core functionality:** Manages form validation, sample inputs, analysis requests, loading/error states, animated score updates, Chart.js rendering, report downloads, and health checks.

**Maintainability notes:** The script uses a small centralized `appState` object. If the dashboard grows, this can be split into API, chart, animation, and report modules.

### `static/css/style.css`

**Purpose:** Provides the visual system and responsive dashboard layout.

**Architecture role:** Styles the HTML template without changing application logic.

**Core functionality:** Defines theme variables, navigation, analyzer card, buttons, loading/error states, result cards, category grids, Chart.js containers, responsive breakpoints, and utility classes.

**Maintainability notes:** CSS custom properties keep colors, spacing, radius, and transitions consistent. Additional components should reuse the existing tokens.

### `requirements.txt`

**Purpose:** Pins Python runtime dependencies.

**Architecture role:** Supports local development, Docker builds, Render deployments, and CI validation.

**Core functionality:** Installs Flask, PyTorch, Hugging Face Transformers, NumPy, dotenv support, Requests, and Gunicorn.

**Maintainability notes:** Pin updates should be tested because model inference libraries can change tokenizer/model behavior between versions.

### `Dockerfile`

**Purpose:** Builds a production container image for the Flask application.

**Architecture role:** Packages the backend, frontend assets, and model runtime dependencies into a deployable unit.

**Core functionality:** Uses Python slim, installs compiler dependencies, installs Python packages, copies source files, exposes port `5000`, and starts Gunicorn.

**Maintainability notes:** For high-traffic deployments, tune Gunicorn workers and timeouts based on CPU, memory, and model load behavior.

### `docker-compose.yml`

**Purpose:** Provides local container orchestration.

**Architecture role:** Runs the web service with production-like environment variables while mapping local source into the container for faster iteration.

**Core functionality:** Builds the local image, maps host port `5000`, sets Flask environment variables, and restarts the service unless stopped.

**Maintainability notes:** Add named volumes for model caches if repeated image rebuilds become slow.

### `.env.example`

**Purpose:** Documents required and optional environment variables.

**Architecture role:** Serves as the template for local `.env` files and hosted platform secrets.

**Core functionality:** Lists Flask environment settings, port configuration, secret key, model references, and optional GPU selection.

**Maintainability notes:** Keep this file synchronized with any new environment variable consumed by `config.py`, deployment manifests, or model loading code.

### Deployment Configuration

**`.github/workflows/deploy.yml`:** Runs CI validation on pushes and pull requests. It installs dependencies, validates Python syntax, runs tests when a `tests/` directory exists, and verifies that the Docker image builds.

**`render.yaml`:** Defines the Render Web Service build and start commands. It installs Python dependencies and starts Gunicorn in production mode.

**`railway.json`:** Configures Railway's Nixpacks builder and Gunicorn start command using Railway's `$PORT` environment variable.

**`Procfile`:** Provides a process command for hosts that use Heroku-style process definitions.

## Deployment

### Docker

```bash
docker build -t toxic-detector:latest .
docker run -p 5000:5000 toxic-detector:latest
```

### Docker Compose

```bash
docker-compose up -d
```

### Render

The included `render.yaml` configures a Python Web Service:

- `buildCommand` installs dependencies from `requirements.txt`.
- `startCommand` runs Gunicorn against `app:app`.
- `FLASK_ENV=production` is set for deployed runtime behavior.

### Railway

The included `railway.json` uses Nixpacks and starts Gunicorn with Railway's injected `$PORT`.

### Production Considerations

- Set a secure `SECRET_KEY`.
- Keep debug mode disabled.
- Add authentication before handling private moderation queues.
- Add rate limiting for public deployments.
- Monitor memory usage during model loading and inference.
- Add request logging and retention policies appropriate to the data being moderated.

## Screenshots

Add dashboard screenshots in a `screenshots/` directory before publishing the repository portfolio page. Recommended captures:

- Main dashboard input state
- Prediction result with category breakdown
- Mobile responsive view

## Future Improvements

- Authentication and role-based access control
- Persistent audit logs with retention controls
- Human review workflow for flagged content
- Dataset-specific evaluation and benchmark reporting
- Environment-driven model selection
- Background processing for larger moderation batches
- CSV and JSON report export
- Automated API tests and frontend smoke tests

## License

MIT License
