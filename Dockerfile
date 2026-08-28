# Serving image: CPU-only, which is what Cloud Run gives you.
# For local GPU TRAINING use an NVIDIA base instead -- see compose.yaml.
FROM python:3.12-slim

# libgomp1 is TensorFlow's OpenMP runtime; the wheel does not bundle it and
# the import fails without it on slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    KERAS_HOME=/opt/keras \
    PORT=8080

WORKDIR /app

# Dependencies first, so edits to our own code do not re-run this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ImageNet weights into the image. Otherwise the first request after
# every cold start pays a ~95 MB download.
RUN python -c "from keras.applications import ResNet50; ResNet50(weights='imagenet', include_top=False)" \
    && chmod -R a+rX /opt/keras

COPY functions/ functions/
COPY api/ api/
# Trained weights when they exist; just .gitkeep otherwise. The predictor
# falls back to untrained and says so in every response.
COPY models/ models/

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8080
# Shell form so Cloud Run's injected $PORT is expanded.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
