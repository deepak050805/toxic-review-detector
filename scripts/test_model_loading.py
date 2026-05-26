"""Standalone transformer pipeline smoke test.

Run this script outside Flask to diagnose Hugging Face cache, proxy, tokenizer,
model, and inference issues independently from the web application.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_loader import get_model, get_model_status


SAMPLES = {
    "positive": "The product arrived quickly and the support team was helpful.",
    "toxic": "You are awful trash and this whole thing is garbage.",
    "neutral": "The package was delivered yesterday and the item is on the table.",
}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    model = get_model()
    print("MODEL_STATUS:", get_model_status())

    for label, text in SAMPLES.items():
        print(f"\n[{label.upper()}] {text}")
        print("toxicity:", model.predict_toxicity(text))
        print("hate_speech:", model.predict_hate_speech(text))
        print("harassment:", model.predict_insult(text))
        print("profanity:", model.predict_profanity(text))
        print("sentiment:", model.predict_sentiment(text))


if __name__ == "__main__":
    main()
