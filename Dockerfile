# Production container for the Flask moderation API and dashboard.
FROM python:3.10-slim

WORKDIR /app

# Compiler packages are required by some Python ML dependencies during install.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependency pins before copying source to improve Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Unbuffered logs are easier to inspect in hosted container platforms.
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Gunicorn provides a production WSGI server around the Flask application.
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 90 --graceful-timeout 30 app:app"]
