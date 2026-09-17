FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git espeak-ng libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml ./
COPY src ./src

ARG EVA_EXTRAS=test
RUN pip install --no-cache-dir ".[${EVA_EXTRAS}]"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    EVA_DATA_DIR=/workspace/.eva

ENTRYPOINT ["python", "-m", "eva.cli"]
