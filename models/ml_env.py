"""Set ML backend environment variables before torch/transformers import.

Must be imported before any Hugging Face or PyTorch modules so optional
TensorFlow backends are never initialized in this PyTorch-only project.
"""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
