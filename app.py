"""Application entry point for the Toxic Review Detector service."""

import logging
import os
import threading
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


def warm_models():
    """Preload transformer pipelines in the background to reduce first-request latency."""
    if model is None:
        return
    try:
        logger.info("Background model warmup started (loading models sequentially)...")
        model._load_models()
        logger.info("Background model warmup finished successfully")
    except Exception:
        logger.exception("Failed to warm up models in background")


threading.Thread(target=warm_models, daemon=True, name="model-warmup").start()

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
    return jsonify({
        "success": False,
        "error": message,
        "model_status": status,
    }), 503


def analyze_text(text):
    """Run toxicity (single pass) and sentiment inference on cleaned text."""
    cleaned_text = text_processor.clean_text(text)

    toxicity_results = model.predict_toxicity_all(cleaned_text)
    sentiment = model.predict_sentiment(cleaned_text)

    return {
        "toxicity": toxicity_results["toxicity"],
        "sentiment": {
            "label": sentiment["label"],
            "confidence": sentiment["confidence"],
            "scores": sentiment.get("scores", {}),
        },
        "categories": toxicity_results["categories"],
    }


@app.route('/')
def index():
    """Render the moderation dashboard."""
    script_path = Path(app.root_path) / 'static' / 'js' / 'script.js'
    asset_version = int(script_path.stat().st_mtime)
    return render_template('index.html', asset_version=asset_version)


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
    try:
        data = _parse_json_body()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400

        text = data.get('text', '').strip()
        is_valid, error_msg = text_processor.validate_text(text)

        if not is_valid:
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400

        if model is None:
            status = get_model_status()
            return jsonify({
                "success": False,
                "error": "Model initialization failed",
                "details": status.get("load_error"),
                "model_status": status,
            }), 500

        if not model.models_ready and model._load_in_progress:
            return _models_unavailable_response()

        response = {
            "success": True,
            **analyze_text(text),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            "Prediction completed: Toxicity=%s (%s%%), Sentiment=%s (%s%%)",
            response["toxicity"]["label"],
            response["toxicity"]["confidence"],
            response["sentiment"]["label"],
            response["sentiment"]["confidence"]
        )
        return jsonify(response), 200

    except ModelLoadError as exc:
        logger.error("Model load error in prediction: %s", exc)
        error_msg = str(exc)
        status_code = 503 if "Timeout" in error_msg else 500
        return jsonify({
            "success": False,
            "error": error_msg
        }), status_code

    except Exception as exc:
        logger.error("Error in prediction: %s", exc, exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(exc)}"
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_predict():
    """Analyze multiple reviews while preserving the single-item score shape."""
    try:
        data = _parse_json_body()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400

        texts = data.get('texts', [])

        if not texts or not isinstance(texts, list):
            return jsonify({
                "success": False,
                "error": "Invalid texts format"
            }), 400

        if len(texts) > 100:
            return jsonify({
                "success": False,
                "error": "Maximum 100 texts per batch"
            }), 400

        if model is None:
            status = get_model_status()
            return jsonify({
                "success": False,
                "error": "Model initialization failed",
                "details": status.get("load_error"),
                "model_status": status,
            }), 500

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

        return jsonify({
            "success": True,
            "count": len(results),
            "results": results
        }), 200

    except ModelLoadError as exc:
        logger.error("Model load error in batch prediction: %s", exc)
        error_msg = str(exc)
        status_code = 503 if "Timeout" in error_msg else 500
        return jsonify({
            "success": False,
            "error": error_msg
        }), status_code

    except Exception as exc:
        logger.error("Error in batch prediction: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(exc)}"
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Return service health and model load status."""
    model_status = get_model_status()
    models_loaded = (
        model is not None
        and model.models_ready
    )
    return jsonify({
        "success": True,
        "status": "healthy" if models_loaded else "degraded",
        "model_loaded": models_loaded,
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Not found"
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "success": False,
        "error": "Request payload is too large"
    }), 413


@app.errorhandler(Exception)
def handle_exception(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({
            "success": False,
            "error": e.description or str(e)
        }), e.code

    logger.exception("Unhandled exception occurred: %s", e)
    return jsonify({
        "success": False,
        "error": f"Internal server error: {str(e)}"
    }), 500


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
