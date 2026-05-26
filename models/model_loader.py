"""Transformer inference layer for moderation scoring.

Loads public Hugging Face pipelines once per process with thread-safe lazy
initialization suitable for Gunicorn single-worker deployments on CPU.
"""

import logging
import os
import threading
import traceback
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

# Public multilabel Jigsaw classifier (toxic, obscene, insult, identity_hate, ...).
TOXICITY_MODEL_ID = "martin-ha/toxic-comment-model"
SENTIMENT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

MAX_INFERENCE_CHARS = 512
LOAD_LOCK_TIMEOUT_SEC = 90.0


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
    explicit_offline = (
        os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
    )
    blocked_proxy_with_cache = has_blackhole_proxy() and model_cache_exists(model_id)
    return explicit_offline or blocked_proxy_with_cache


def _score_by_label(results, label, default=0.0):
    """Read a class probability from pipeline output (case-insensitive)."""
    target = label.lower()
    for item in results:
        if str(item.get("label", "")).lower() == target:
            return float(item.get("score", default))
    return default


def _normalize_sentiment_scores(results):
    """Map SST-2 style outputs to positive/negative/neutral percentages."""
    scores = {
        str(item["label"]).lower(): round(float(item["score"]) * 100, 1)
        for item in results
    }

    positive = scores.get("positive", scores.get("label_1", 0.0))
    negative = scores.get("negative", scores.get("label_0", 0.0))
    neutral = scores.get("neutral", 0.0)

    if neutral == 0.0 and "neutral" not in scores:
        neutral = round(max(0.0, 100.0 - positive - negative), 1)

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
    }


class ToxicityModelLoader:
    """Load and reuse transformer pipelines for moderation categories."""

    def __init__(self):
        self.toxicity_model = None
        self.sentiment_model = None
        self._lock = threading.Lock()
        self._load_in_progress = False
        self.status = {
            "device": "cpu",
            "cache_path": str(get_hf_cache_path()),
            "models": {},
            "load_error": None,
        }
        logger.info("Lazy loading enabled for transformer pipelines (CPU-only)")

    @property
    def models_ready(self):
        """True when both toxicity and sentiment pipelines are in memory."""
        return self.toxicity_model is not None and self.sentiment_model is not None

    def _load_pipeline(self, label, model_id):
        """Load a text-classification pipeline with deployment-safe defaults."""
        local_files_only = should_use_local_files(model_id)
        self.status["models"][label] = {
            "model_id": model_id,
            "local_files_only": local_files_only,
            "loaded": False,
        }

        logger.info("[%s] Device: cpu", label)
        logger.info("[%s] Cache path: %s", label, self.status["cache_path"])
        logger.info("[%s] Loading model: %s", label, model_id)
        logger.info("[%s] Local cache available: %s", label, model_cache_exists(model_id))
        logger.info("[%s] local_files_only=%s", label, local_files_only)

        hub_kwargs = {"local_files_only": local_files_only}
        pipeline_kwargs = {
            "task": "text-classification",
            "model": model_id,
            "tokenizer": model_id,
            "device": -1,
            "top_k": 4,
            "model_kwargs": hub_kwargs,
        }
        if label == "Toxicity":
            pipeline_kwargs["function_to_apply"] = "sigmoid"

        try:
            inference_pipeline = pipeline(**pipeline_kwargs)
            self.status["models"][label]["loaded"] = True
            logger.info("[%s] Pipeline initialized successfully", label)
            return inference_pipeline

        except Exception as exc:
            self.status["models"][label]["error"] = str(exc)
            self.status["load_error"] = f"{label} failed: {exc}"
            logger.exception("[%s] Failed to initialize pipeline for %s", label, model_id)
            raise ModelLoadError(f"{label} model failed to load: {exc}") from exc

    def _load_models(self):
        """Load all pipelines sequentially while holding the process lock."""
        logger.info("Starting transformer moderation pipeline initialization")
        logger.info("PyTorch version: %s", torch.__version__)
        logger.info("Proxy blocked: %s", has_blackhole_proxy())

        with self._lock:
            self._load_in_progress = True
            try:
                if self.toxicity_model is None:
                    self.toxicity_model = self._load_pipeline("Toxicity", TOXICITY_MODEL_ID)
                if self.sentiment_model is None:
                    self.sentiment_model = self._load_pipeline("Sentiment", SENTIMENT_MODEL_ID)
            finally:
                self._load_in_progress = False

        logger.info("All transformer moderation pipelines loaded successfully")

    def _ensure_pipeline(self, attr_name, label, model_id):
        """Load a single pipeline with timeout-safe double-checked locking."""
        if getattr(self, attr_name) is not None:
            return

        acquired = self._lock.acquire(timeout=LOAD_LOCK_TIMEOUT_SEC)
        if not acquired:
            raise ModelLoadError(
                f"Timeout waiting for {label} model loading to complete."
            )

        try:
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, self._load_pipeline(label, model_id))
        finally:
            self._lock.release()

    def _run_inference(self, pipe, text):
        """Run one forward pass with inference mode enabled."""
        snippet = text[:MAX_INFERENCE_CHARS]
        with torch.inference_mode():
            output = pipe(snippet)
        return output[0] if output else []

    def predict_toxicity_all(self, text):
        """Single toxicity forward pass; derive all dashboard category scores."""
        self._ensure_pipeline("toxicity_model", "Toxicity", TOXICITY_MODEL_ID)

        if not text or not text.strip():
            return {
                "toxicity": {"label": "Clean", "confidence": 0.0},
                "categories": {
                    "toxicity": 0.0,
                    "hate_speech": 0.0,
                    "harassment": 0.0,
                    "profanity": 0.0,
                },
            }

        try:
            results = self._run_inference(self.toxicity_model, text)
            toxic_score = _score_by_label(results, "toxic")
            hate_score = _score_by_label(results, "identity_hate")
            obscene_score = _score_by_label(results, "obscene")
            insult_score = _score_by_label(results, "insult")

            toxic_conf = round(toxic_score * 100, 1)
            return {
                "toxicity": {
                    "label": "Toxic" if toxic_score > 0.5 else "Clean",
                    "confidence": toxic_conf,
                },
                "categories": {
                    "toxicity": toxic_conf,
                    "hate_speech": round(hate_score * 100, 1),
                    "harassment": round(insult_score * 100, 1),
                    "profanity": round(obscene_score * 100, 1),
                },
            }
        except Exception:
            logger.exception("Error in toxicity prediction")
            return {
                "toxicity": {"label": "Error", "confidence": 0.0},
                "categories": {
                    "toxicity": 0.0,
                    "hate_speech": 0.0,
                    "harassment": 0.0,
                    "profanity": 0.0,
                },
            }

    def predict_sentiment(self, text):
        """Return sentiment label, confidence, and normalized score distribution."""
        self._ensure_pipeline("sentiment_model", "Sentiment", SENTIMENT_MODEL_ID)

        if not text or not text.strip():
            return {
                "label": "Neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.0, "neutral": 100.0, "negative": 0.0},
            }

        try:
            results = self._run_inference(self.sentiment_model, text)
            scores = _normalize_sentiment_scores(results)
            best_match = max(results, key=lambda item: item["score"])
            label = str(best_match["label"]).capitalize()

            return {
                "label": label,
                "confidence": round(float(best_match["score"]) * 100, 1),
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
_model_singleton_lock = threading.Lock()


def get_model():
    """Return the process-wide model loader instance."""
    global _model_instance, _model_load_error, _model_load_traceback

    if _model_instance is not None:
        return _model_instance

    with _model_singleton_lock:
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
            "loaded": _model_instance.models_ready,
            "warming": _model_instance._load_in_progress,
            **_model_instance.status,
        }

    return {
        "loaded": False,
        "warming": False,
        "device": "cpu",
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
