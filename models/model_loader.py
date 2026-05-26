"""Transformer inference layer for moderation scoring.

The Flask API depends on this module for all model interaction. It wraps
Hugging Face text-classification pipelines behind a small application-specific
interface so route handlers do not need to know about tokenizer limits, device
selection, cache state, or raw model labels.
"""

import logging
import os
import traceback
from pathlib import Path

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

TOXICITY_MODEL_ID = "martin-ha/toxic-comment-model"
SENTIMENT_MODEL_ID = "lxyuan/distilbert-base-uncased-finetuned-sst-2-english"


class ModelLoadError(RuntimeError):
    """Raised when a required transformer pipeline cannot be initialized."""


def get_hf_cache_path():
    """Return the Hugging Face cache path used by transformers/huggingface_hub."""
    explicit_cache = (
        os.environ.get("HF_HOME")
        or os.environ.get("TRANSFORMERS_CACHE")
        or os.environ.get("HF_HUB_CACHE")
    )

    if explicit_cache:
        return Path(explicit_cache).expanduser()

    return Path.home() / ".cache" / "huggingface" / "hub"


def has_blackhole_proxy():
    """Detect local proxy settings that intentionally block outbound requests."""
    proxy_vars = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")
    return any("127.0.0.1:9" in os.environ.get(name, "") for name in proxy_vars)


def model_cache_exists(model_id):
    """Return whether the requested model has a local Hugging Face cache folder."""
    cache_name = f"models--{model_id.replace('/', '--')}"
    return (get_hf_cache_path() / cache_name).exists()


def should_use_local_files(model_id):
    """Prefer cached files when offline mode or a blocked local proxy is active."""
    explicit_offline = os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"
    blocked_proxy_with_cache = has_blackhole_proxy() and model_cache_exists(model_id)
    return explicit_offline or blocked_proxy_with_cache


class ToxicityModelLoader:
    """Load and reuse transformer pipelines for moderation categories.

    A single loader instance is kept for the process lifetime because model
    initialization is expensive. The class exposes score-oriented methods that
    return application-friendly values instead of raw pipeline outputs.
    """

    def __init__(self):
        self.toxicity_model = None
        self.sentiment_model = None
        self.status = {
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "cache_path": str(get_hf_cache_path()),
            "models": {},
            "load_error": None,
        }
        logger.info("Lazy loading enabled for transformer pipelines")

    def _load_pipeline(self, label, model_id):
        """Load a lightweight pipeline directly from model_id with detailed startup logging."""
        local_files_only = should_use_local_files(model_id)
        self.status["models"][label] = {
            "model_id": model_id,
            "local_files_only": local_files_only,
            "loaded": False,
        }

        logger.info("[%s] Device selected: cpu (forced for low-memory deployment)", label)
        logger.info("[%s] Hugging Face cache path: %s", label, self.status["cache_path"])
        logger.info("[%s] Loading model: %s", label, model_id)
        logger.info("[%s] Local cache available: %s", label, model_cache_exists(model_id))
        logger.info("[%s] local_files_only=%s", label, local_files_only)

        try:
            logger.info("[%s] Initializing lightweight pipeline...", label)

            inference_pipeline = pipeline(
                "text-classification",
                model=model_id,
                device=-1,
                top_k=None,
            )

            logger.info("[%s] Pipeline initialized successfully", label)

            self.status["models"][label]["loaded"] = True

            return inference_pipeline

        except Exception as exc:
            self.status["models"][label]["error"] = str(exc)
            self.status["load_error"] = f"{label} failed: {exc}"

            logger.exception(
                "[%s] Failed to initialize pipeline for %s",
                label,
                model_id,
            )

            raise ModelLoadError(
                f"{label} model failed to load: {exc}"
            ) from exc

    def _load_models(self):
        """Load all required pretrained text-classification pipelines."""
        logger.info("Starting transformer moderation pipeline initialization")
        logger.info("PyTorch version: %s", torch.__version__)
        logger.info("CUDA available: %s", torch.cuda.is_available())
        logger.info("Proxy detected as blocked: %s", has_blackhole_proxy())

        self.toxicity_model = self._load_pipeline("Toxicity", TOXICITY_MODEL_ID)
        self.sentiment_model = self._load_pipeline("Sentiment", SENTIMENT_MODEL_ID)

        logger.info("All transformer moderation pipelines loaded successfully")

    def predict_toxicity(self, text):
        """Return the primary toxicity label and score on a 0-100 scale."""
        if self.toxicity_model is None:
            self.toxicity_model = self._load_pipeline(
                "Toxicity",
                TOXICITY_MODEL_ID,
            )

        if not text or len(text.strip()) == 0:
            return {"label": "Clean", "score": 0.0}

        try:
            results = self.toxicity_model(text[:512])[0]
            toxic_score = next((item["score"] for item in results if item["label"] == "toxic"), 0.0)

            return {
                "label": "Toxic" if toxic_score > 0.5 else "Clean",
                "score": round(toxic_score * 100, 1),
            }
        except Exception:
            logger.exception("Error in toxicity prediction")
            return {"label": "Error", "score": 0.0}

    def predict_hate_speech(self, text):
        """Return an identity-hate indicator score on a 0-100 scale."""
        if not text or len(text.strip()) == 0:
            return 0.0

        try:
            results = self.toxicity_model(text[:512])[0]
            hate_score = next((item["score"] for item in results if item["label"] == "identity_hate"), 0.0)
            return round(hate_score * 100, 1)
        except Exception:
            logger.exception("Error in hate speech detection")
            return 0.0

    def predict_profanity(self, text):
        """Return an obscene/profanity score on a 0-100 scale."""
        if not text or len(text.strip()) == 0:
            return 0.0

        try:
            results = self.toxicity_model(text[:512])[0]
            obscene_score = next((item["score"] for item in results if item["label"] == "obscene"), 0.0)
            return round(obscene_score * 100, 1)
        except Exception:
            logger.exception("Error in profanity detection")
            return 0.0

    def predict_insult(self, text):
        """Return an insult score on a 0-100 scale, representing harassment."""
        if not text or len(text.strip()) == 0:
            return 0.0

        try:
            results = self.toxicity_model(text[:512])[0]
            insult_score = next((item["score"] for item in results if item["label"] == "insult"), 0.0)
            return round(insult_score * 100, 1)
        except Exception:
            logger.exception("Error in insult detection")
            return 0.0

    def predict_sentiment(self, text):
        """Return sentiment label, confidence, and class score distribution."""
        if self.sentiment_model is None:
            self.sentiment_model = self._load_pipeline(
                "Sentiment",
                SENTIMENT_MODEL_ID,
            )

        if not text or len(text.strip()) == 0:
            return {
                "label": "Neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.0, "neutral": 100.0, "negative": 0.0},
            }

        try:
            results = self.sentiment_model(text[:512])[0]
            scores = {item["label"].lower(): round(item["score"] * 100, 1) for item in results}
            best_match = max(results, key=lambda item: item["score"])

            return {
                "label": best_match["label"].capitalize(),
                "confidence": round(best_match["score"] * 100, 1),
                "scores": scores,
            }
        except Exception:
            logger.exception("Error in sentiment prediction")
            return {
                "label": "Neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.0, "neutral": 100.0, "negative": 0.0},
            }


_model_instance = None
_model_load_error = None
_model_load_traceback = None


def get_model():
    """Return the process-wide model loader instance."""
    global _model_instance, _model_load_error, _model_load_traceback
    if _model_instance is None:
        try:
            _model_instance = ToxicityModelLoader()
            _model_load_error = None
            _model_load_traceback = None
        except Exception as exc:
            _model_load_error = str(exc)
            _model_load_traceback = traceback.format_exc()
            raise
    return _model_instance


def get_model_status():
    """Return model availability details for health checks and diagnostics."""
    if _model_instance is not None:
        return {
            "loaded": True,
            **_model_instance.status,
        }

    return {
        "loaded": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "cache_path": str(get_hf_cache_path()),
        "proxy_blocked": has_blackhole_proxy(),
        "load_error": _model_load_error,
        "traceback": _model_load_traceback,
        "models": {
            "Toxicity": {
                "model_id": TOXICITY_MODEL_ID,
                "cache_available": model_cache_exists(TOXICITY_MODEL_ID),
            },
            "Sentiment": {
                "model_id": SENTIMENT_MODEL_ID,
                "cache_available": model_cache_exists(SENTIMENT_MODEL_ID),
            },
        },
    }