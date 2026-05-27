"""Set ML backend environment variables before torch/transformers import.

Must be imported before any Hugging Face or PyTorch modules so optional
TensorFlow backends are never initialized in this PyTorch-only project.
"""

import os

# Disable TensorFlow entirely to save startup time and RAM
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Ensure CPU-only inference (Render doesn't have GPUs)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCH_DEVICE", "cpu")

# Disable parallelization that causes contention in Gunicorn single-worker mode
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Suppress verbose library logging during startup
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

