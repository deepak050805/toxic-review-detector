"""Application entry point for the Toxic Review Detector service.

This module owns the HTTP boundary of the moderation platform. It renders the
dashboard template, validates API payloads, invokes preprocessing and model
inference, and returns normalized JSON responses for frontend and API clients.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from models.model_loader import get_model, get_model_status
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

# Models are loaded at process startup so the first prediction request does not
# need to pay initialization cost. Startup failures are surfaced through API
# responses instead of crashing the web process.
try:
    model = get_model()
    logger.info("NLP models loaded successfully")
except Exception as exc:
    logger.exception("Failed to load NLP models")
    model = None

text_processor = TextProcessor()


def analyze_text(text):
    """Run the configured moderation models and return normalized scores.

    Toxicity, hate speech, profanity, and insult are scored using unitary/toxic-bert,
    while sentiment is scored using lxyuan/distilbert-base-multilingual-cased-sentiments-student.
    """
    cleaned_text = text_processor.clean_text(text)
    
    toxicity = model.predict_toxicity(cleaned_text)
    hate_speech = model.predict_hate_speech(cleaned_text)
    profanity = model.predict_profanity(cleaned_text)
    harassment = model.predict_insult(cleaned_text)
    sentiment = model.predict_sentiment(cleaned_text)

    return {
        "toxicity": {
            "label": toxicity['label'],
            "confidence": toxicity['score']
        },
        "sentiment": {
            "label": sentiment['label'],
            "confidence": sentiment['confidence'],
            "scores": sentiment.get('scores', {})
        },
        "categories": {
            "toxicity": toxicity['score'],
            "hate_speech": hate_speech,
            "harassment": harassment,
            "profanity": profanity
        }
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
        data = request.get_json()

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
        data = request.get_json()

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

        results = []
        for text in texts:
            normalized_text = str(text).strip()
            is_valid, _ = text_processor.validate_text(normalized_text)

            if not is_valid:
                continue

            result = analyze_text(normalized_text)
            # Return a short text preview so batch responses remain compact.
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

    except Exception as exc:
        logger.error("Error in batch prediction: %s", exc)
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Return service health and model load status."""
    model_status = get_model_status()
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "success": False,
        "error": "Request payload is too large"
    }), 413


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
