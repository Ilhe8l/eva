FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git espeak-ng libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

ENV UV_PROJECT_ENVIRONMENT=/opt/eva \
    PATH=/opt/eva/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

ARG EVA_EXTRAS="--extra test"
RUN uv sync --frozen ${EVA_EXTRAS}

ENTRYPOINT ["eva"]
