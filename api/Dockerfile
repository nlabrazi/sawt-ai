# syntax=docker/dockerfile:1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ARG PYTORCH_INDEX_URL
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && TORCH_PKGS="$(grep -E '^(torch|torchaudio)==' requirements.txt || true)" \
    && if [ -n "$TORCH_PKGS" ]; then \
         pip install --index-url "${PYTORCH_INDEX_URL}" $TORCH_PKGS; \
       fi \
    && pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
