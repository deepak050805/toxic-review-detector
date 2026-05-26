"""Application entry point for the Toxic Review Detector service."""

import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import models.ml_env  # noqa: F401 — disable TensorFlow before ML imports

from models.model_loader import get_model, get_model_status, ModelLoadError
from utils.text_processing import TextProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug-6cebb5.log"

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

try:
    model = get_model()
    logger.info("NLP models loader initialized successfully")
except Exception:
    logger.exception("Failed to initialize NLP models loader")
    model = None


# region agent log
def _agent_log(hypothesis_id, location, message, data=None, run_id="run1"):
    payload = {
        "sessionId": "6cebb5",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass
# endregion


def _log_event(event, hypothesis_id="H0", **fields):
    """Structured deployment/inference logging."""
    logger.info("event=%s %s", event, fields)
    # region agent log
    _agent_log(hypothesis_id, "app.py:_log_event", event, {"event": event, **fields})
    # endregion


def _json_error(message, status_code=500, error_type="internal_error", extra=None):
    """Always return a non-empty JSON error body."""
    body = {
        "success": False,
        "error": message or "Internal server error",
        "error_type": error_type,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        body.update(extra)
    response = jsonify(body)
    response.status_code = status_code
    response.headers["Content-Type"] = "application/json"
    return response


def warm_models():
    """Preload transformer pipelines in the background to reduce first-request latency."""
    if model is None:
        _log_event("model_warmup_skipped", hypothesis_id="H4", reason="loader_none")
        return
    _log_event("model_warmup_started", hypothesis_id="H4")
    try:
        model._load_models()
        _log_event("model_warmup_completed", hypothesis_id="H4", models_ready=model.models_ready)
    except MemoryError as exc:
        _log_event("model_warmup_oom", hypothesis_id="H3", error=str(exc))
        logger.exception("OOM during model warmup")
    except Exception as exc:
        _log_event("model_warmup_failed", hypothesis_id="H4", error=str(exc))
        logger.exception("Failed to warm up models in background")


# threading.Thread(target=warm_models, daemon=True, name="model-warmup").start()

text_processor = TextProcessor()


def _parse_json_body():
    """Parse JSON bodies without raising HTML 400 pages on malformed input."""
    return request.get_json(silent=True)


def _models_unavailable_response():
    """Return a consistent JSON payload when inference is not ready."""
    status = get_model_status()
    message = "Moderation models are still loading. Please retry in a moment."
    if status.get("load_error"):
        message = f"Model initialization failed: {status['load_error']}"
    return _json_error(
        message,
        status_code=503,
        error_type="models_unavailable",
        extra={"model_status": status},
    )


def analyze_text(text):
    """Run toxicity (single pass) and sentiment inference on cleaned text."""
    _log_event("inference_started", hypothesis_id="H2", text_len=len(text))
    try:
        cleaned_text = text_processor.clean_text(text)
        toxicity_results = model.predict_toxicity_all(cleaned_text)
        sentiment = model.predict_sentiment(cleaned_text)
        result = {
            "toxicity": toxicity_results["toxicity"],
            "sentiment": {
                "label": sentiment["label"],
                "confidence": sentiment["confidence"],
                "scores": sentiment.get("scores", {}),
            },
            "categories": toxicity_results["categories"],
        }
        _log_event(
            "inference_completed",
            hypothesis_id="H2",
            toxicity_label=result["toxicity"]["label"],
            sentiment_label=result["sentiment"]["label"],
        )
        return result
    except MemoryError:
        _log_event("inference_oom", hypothesis_id="H3")
        raise
    except Exception:
        _log_event("inference_failed", hypothesis_id="H2")
        raise


@app.route('/')
def index():
    """Render the moderation dashboard."""
    script_path = Path(app.root_path) / 'static' / 'js' / 'script.js'
    asset_version = int(script_path.stat().st_mtime)
    return render_template('index.html', asset_version=asset_version)


@app.after_request
def ensure_api_json_response(response):
    """Guarantee /api routes never return an empty body (H5)."""
    if request.path.startswith("/api/"):
        response.headers.setdefault("Content-Type", "application/json")
        if not response.get_data():
            # region agent log
            _agent_log("H5", "app.py:ensure_api_json_response", "empty_body_replaced", {
                "path": request.path,
                "status": response.status_code,
            })
            # endregion
            return _json_error(
                "Server returned an empty response",
                status_code=response.status_code or 500,
                error_type="empty_response",
            )
    return response


@app.after_request
def disable_static_cache(response):
    """Prevent stale JavaScript from persisting during local development."""
    if request.endpoint == 'static':
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.route('/api/predict', methods=['POST'])
def predict():
    """Analyze one review and return the dashboard/API response payload."""
    # region agent log
    _agent_log("H2", "app.py:predict", "predict_entered", {"path": request.path})
    # endregion
    try:
        return _predict_impl()
    except MemoryError as exc:
        _log_event("predict_oom", hypothesis_id="H3", error=str(exc))
        return _json_error(
            "Inference failed due to insufficient memory",
            status_code=503,
            error_type="oom",
        )
    except ModelLoadError as exc:
        error_msg = str(exc)
        if "Timeout" in error_msg:
            _log_event("predict_timeout", hypothesis_id="H1", error=error_msg)
            return _json_error(error_msg, status_code=503, error_type="timeout")
        _log_event("predict_model_load_error", hypothesis_id="H2", error=error_msg)
        return _json_error(error_msg, status_code=500, error_type="model_load")
    except Exception as exc:
        _log_event("predict_fatal", hypothesis_id="H2", error=str(exc), tb=traceback.format_exc()[:500])
        logger.exception("Fatal error in /api/predict")
        return _json_error(
            f"Inference failed: {exc}",
            status_code=500,
            error_type="inference_error",
        )


def _predict_impl():
    """Inner predict handler; exceptions bubble to predict() for JSON responses."""
    data = _parse_json_body()

    if not data:
        return _json_error("No JSON data provided", status_code=400, error_type="validation")

    text = data.get('text', '').strip()
    is_valid, error_msg = text_processor.validate_text(text)

    if not is_valid:
        return _json_error(error_msg, status_code=400, error_type="validation")

    if model is None:
        status = get_model_status()
        return _json_error(
            "Model initialization failed",
            status_code=500,
            error_type="model_init",
            extra={"details": status.get("load_error"), "model_status": status},
        )

    if not model.models_ready and model._load_in_progress:
        return _models_unavailable_response()

    response = {
        "success": True,
        **analyze_text(text),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        "Prediction completed: Toxicity=%s (%s%%), Sentiment=%s (%s%%)",
        response["toxicity"]["label"],
        response["toxicity"]["confidence"],
        response["sentiment"]["label"],
        response["sentiment"]["confidence"],
    )
    out = jsonify(response)
    out.headers["Content-Type"] = "application/json"
    return out, 200


@app.route('/api/batch', methods=['POST'])
def batch_predict():
    """Analyze multiple reviews while preserving the single-item score shape."""
    try:
        data = _parse_json_body()

        if not data:
            return _json_error("No JSON data provided", status_code=400, error_type="validation")

        texts = data.get('texts', [])

        if not texts or not isinstance(texts, list):
            return _json_error("Invalid texts format", status_code=400, error_type="validation")

        if len(texts) > 100:
            return _json_error("Maximum 100 texts per batch", status_code=400, error_type="validation")

        if model is None:
            status = get_model_status()
            return _json_error(
                "Model initialization failed",
                status_code=500,
                error_type="model_init",
                extra={"details": status.get("load_error"), "model_status": status},
            )

        if not model.models_ready and model._load_in_progress:
            return _models_unavailable_response()

        results = []
        for text in texts:
            normalized_text = str(text).strip()
            is_valid, _ = text_processor.validate_text(normalized_text)

            if not is_valid:
                continue

            result = analyze_text(normalized_text)
            result["text"] = (
                f"{normalized_text[:100]}..."
                if len(normalized_text) > 100
                else normalized_text
            )
            results.append(result)

        out = jsonify({"success": True, "count": len(results), "results": results})
        out.headers["Content-Type"] = "application/json"
        return out, 200

    except MemoryError as exc:
        _log_event("batch_oom", hypothesis_id="H3", error=str(exc))
        return _json_error(
            "Batch inference failed due to insufficient memory",
            status_code=503,
            error_type="oom",
        )
    except ModelLoadError as exc:
        error_msg = str(exc)
        status_code = 503 if "Timeout" in error_msg else 500
        error_type = "timeout" if "Timeout" in error_msg else "model_load"
        if error_type == "timeout":
            _log_event("batch_timeout", hypothesis_id="H1", error=error_msg)
        return _json_error(error_msg, status_code=status_code, error_type=error_type)
    except Exception as exc:
        logger.exception("Error in batch prediction: %s", exc)
        return _json_error(
            f"Internal server error: {exc}",
            status_code=500,
            error_type="inference_error",
        )


@app.route('/api/health', methods=['GET'])
def health():
    """Return service health and model load status."""
    model_status = get_model_status()
    models_loaded = model is not None and model.models_ready
    out = jsonify({
        "success": True,
        "status": "healthy" if models_loaded else "degraded",
        "model_loaded": models_loaded,
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
    })
    out.headers["Content-Type"] = "application/json"
    return out, 200


@app.errorhandler(404)
def not_found(error):
    return _json_error("Not found", status_code=404, error_type="not_found")


@app.errorhandler(500)
def server_error(error):
    return _json_error("Internal server error", status_code=500, error_type="internal_error")


@app.errorhandler(413)
def request_too_large(error):
    return _json_error("Request payload is too large", status_code=413, error_type="payload_too_large")


@app.errorhandler(Exception)
def handle_exception(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return _json_error(e.description or str(e), status_code=e.code, error_type="http_error")

    logger.exception("Unhandled exception occurred: %s", e)
    return _json_error(
        f"Internal server error: {e}",
        status_code=500,
        error_type="unhandled",
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    logger.info("Starting Toxic Review Detector on port %s", port)
    logger.info("Debug mode: %s", debug)

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=debug
    )
