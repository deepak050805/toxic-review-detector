# Project Structure

This document describes the role of each important file in the Toxic Review Detector codebase. It is intended as a developer reference for understanding how the Flask API, transformer inference layer, frontend dashboard, and deployment files fit together.

## Architecture Summary

```text
Browser Dashboard
        |
        v
static/js/script.js
        |
        v
Flask routes in app.py
        |
        v
utils/text_processing.py
        |
        v
models/model_loader.py
        |
        v
Hugging Face Transformers + PyTorch
        |
        v
JSON response rendered by the dashboard
```

## Root Files

### `app.py`

**Purpose:** Main Flask application entry point.

**Architecture role:** Connects the web dashboard, API endpoints, preprocessing utilities, and model inference layer.

**Core functionality:**

- Renders `templates/index.html`.
- Provides `/api/health`, `/api/predict`, and `/api/batch`.
- Validates request payloads.
- Calls `TextProcessor` before model inference.
- Calls the singleton transformer loader from `models/model_loader.py`.
- Returns consistent JSON success and error responses.

**Maintainability notes:** The shared `analyze_text()` helper keeps prediction logic consistent across single and batch workflows.

### `config.py`

**Purpose:** Centralized runtime configuration.

**Architecture role:** Keeps environment-specific Flask settings outside route handlers.

**Core functionality:**

- Loads `.env` values through `python-dotenv`.
- Defines development, production, and testing config classes.
- Provides `get_config()` for future application factory usage.

**Maintainability notes:** Add new environment-driven settings here instead of reading environment variables throughout the application.

### `requirements.txt`

**Purpose:** Defines Python dependency pins.

**Architecture role:** Used by local setup, Docker, Render, Railway, and CI.

**Core functionality:** Installs Flask, Gunicorn, PyTorch, Hugging Face Transformers, NumPy, Requests, and dotenv support.

**Maintainability notes:** Update ML dependencies carefully because tokenizer and pipeline behavior can change across versions.

### `README.md`

**Purpose:** Primary repository documentation.

**Architecture role:** Explains product scope, system architecture, request lifecycle, model usage, API contracts, and deployment strategy.

**Maintainability notes:** Keep endpoint examples and environment variables synchronized with code changes.

### `.env.example`

**Purpose:** Documents environment variables.

**Architecture role:** Serves as a safe template for local `.env` files and hosted platform secrets.

**Maintainability notes:** Never commit real secrets. Add new variables here when code or deployment manifests require them.

## Backend Package

### `models/model_loader.py`

**Purpose:** Transformer model loading and scoring.

**Architecture role:** Encapsulates Hugging Face and PyTorch usage behind application-specific methods.

**Core functionality:**

- Selects CUDA when available and CPU otherwise.
- Loads toxicity, hate-speech, and abusive-language pipelines.
- Converts raw model outputs to normalized `0-100` scores.
- Reuses a process-wide singleton model instance.

**Maintainability notes:** Move model names into configuration if different environments need different model sizes or languages.

### `models/__init__.py`

**Purpose:** Marks `models` as an importable Python package.

**Architecture role:** Allows `app.py` to import model utilities using package-style imports.

**Maintainability notes:** Keep package initialization minimal unless shared model package exports become useful.

### `utils/text_processing.py`

**Purpose:** Text preprocessing and validation.

**Architecture role:** Sits between API input handling and transformer inference.

**Core functionality:**

- Removes URLs and email addresses.
- Normalizes whitespace.
- Validates minimum and maximum review length.
- Provides keyword extraction support for future reporting workflows.

**Maintainability notes:** This module is framework-independent and should be covered by unit tests first.

### `utils/__init__.py`

**Purpose:** Marks `utils` as an importable Python package.

**Architecture role:** Supports clean imports from route handlers and future backend modules.

## Frontend Files

### `templates/index.html`

**Purpose:** Server-rendered dashboard template.

**Architecture role:** Defines the DOM structure consumed by `static/js/script.js` and styled by `static/css/style.css`.

**Core functionality:**

- Navigation and product summary.
- Review submission form.
- Loading and error states.
- Result summary card.
- Category score cards.
- Chart containers.
- Processing flow and API reference summaries.

**Maintainability notes:** IDs used by JavaScript are part of the frontend contract. Update selectors and markup together.

### `static/js/script.js`

**Purpose:** Browser-side dashboard controller.

**Architecture role:** Connects user interactions to Flask API endpoints through the Fetch API.

**Core functionality:**

- Tracks dashboard state.
- Validates review length.
- Sends `/api/predict` requests.
- Handles API errors and loading states.
- Animates scores and progress bars.
- Renders Chart.js bar and doughnut charts.
- Generates plain-text moderation reports.
- Checks backend health on page load.

**Maintainability notes:** If the UI grows, split API, charts, animation, and report generation into separate modules.

### `static/css/style.css`

**Purpose:** Dashboard visual system.

**Architecture role:** Defines responsive presentation while keeping layout and behavior separate from JavaScript.

**Core functionality:**

- Theme variables for color, spacing, radius, typography, and transitions.
- Navigation, hero, analyzer, result, category, chart, and footer styles.
- Loading/error states.
- Responsive breakpoints for tablet and mobile layouts.

**Maintainability notes:** Reuse existing CSS custom properties when adding new UI sections.

## Deployment and Operations

### `Dockerfile`

**Purpose:** Builds a deployable Python container image.

**Architecture role:** Packages application code, frontend assets, and ML runtime dependencies for container platforms.

**Core functionality:**

- Uses `python:3.10-slim`.
- Installs compiler dependencies required by ML packages.
- Installs Python dependencies from `requirements.txt`.
- Starts Flask through Gunicorn.

**Maintainability notes:** Tune worker count and timeout for the deployment target and model load time.

### `docker-compose.yml`

**Purpose:** Local container orchestration.

**Architecture role:** Provides a production-like runtime for local testing.

**Core functionality:**

- Builds the application image.
- Maps port `5000`.
- Sets production-style environment variables.
- Mounts local source into the container for development iteration.

**Maintainability notes:** Add named cache volumes for Hugging Face model artifacts if local container rebuilds become slow.

### `render.yaml`

**Purpose:** Render Web Service configuration.

**Architecture role:** Defines build/start commands and production environment values for Render.

**Core functionality:** Installs dependencies and starts Gunicorn with `app:app`.

**Maintainability notes:** Move sensitive values to the Render dashboard rather than committing them.

### `railway.json`

**Purpose:** Railway deployment configuration.

**Architecture role:** Uses Railway's Nixpacks builder and starts Gunicorn against the Flask app.

**Core functionality:** References Railway's injected `$PORT` variable and restart policy settings.

**Maintainability notes:** Keep the start command aligned with `Dockerfile`, `render.yaml`, and `Procfile`.

### `Procfile`

**Purpose:** Process definition for Heroku-style Python hosts.

**Architecture role:** Provides a minimal web process command for platforms that read Procfiles.

**Core functionality:** Starts Gunicorn with two workers and binds to the platform-provided `$PORT`.

### `.github/workflows/deploy.yml`

**Purpose:** Continuous integration workflow.

**Architecture role:** Validates the repository on pushes and pull requests before deployment.

**Core functionality:**

- Installs dependencies.
- Runs Python syntax validation.
- Runs tests when a `tests/` directory exists.
- Builds the Docker image.

**Maintainability notes:** Add real unit and API tests, then make the test step required for pull requests.

## Documentation Files

### `GETTING_STARTED.md`

**Purpose:** Practical setup guide for local, Docker, and basic API usage.

### `QUICK_START.md`

**Purpose:** Short command reference for running the app and calling the API.

### `DEPLOYMENT_GUIDE.md`

**Purpose:** Deployment-oriented guide for Docker, Render, Railway, and production checks.

## Recommended Scalability Path

1. Move model identifiers and thresholds into configuration.
2. Add automated tests for preprocessing and API response contracts.
3. Add authentication and rate limiting for public deployments.
4. Add asynchronous batch processing for larger workloads.
5. Add persistent audit logs and reviewer workflow states.
6. Benchmark models on a representative dataset before publishing performance claims.
