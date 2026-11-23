# syntax=docker/dockerfile:1
#
# One self-contained image: FastAPI + hybrid retriever (bge embeddings, Chroma,
# BM25) + the static UI. The LLM is external - a hosted OpenAI-compatible API by
# default, or Ollama (see docker-compose.yml) - because a 4B model on a free
# CPU-only host is most of a minute per answer.
#
# Build:  docker build -t uk-rental-law-rag .
# Run:    docker run -p 7860:7860 -e GROQ_API_KEY=... uk-rental-law-rag
#
# The corpus is the committed legislation-only build (data/processed/): Crown
# copyright under the Open Government Licence. Case law is deliberately not in
# the image - see README, Known limitations, on the Find Case Law licence.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch, installed first and on its own layer: the default PyPI wheel
# bundles ~2.5GB of CUDA libraries a CPU host never loads. Same version the
# eval baselines were measured with (see constraints.txt).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# Bake the embedding model into the image so a container start never downloads
# it - a cold start is already the slowest moment for a visitor.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Build the index at image build time from the committed chunks: the container
# starts with a ready index and needs no write access to build one.
COPY backend/ backend/
COPY ingestion/build_index.py ingestion/build_index.py
COPY data/processed/ data/processed/
RUN python ingestion/build_index.py

COPY frontend/ frontend/
COPY eval/ eval/

# Run as an unprivileged user that owns /app: Chroma opens its SQLite file
# read-write even for queries.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# Public-hosting defaults; override with -e for local use. The API key is never
# baked in - pass it at run time (or as the host's secret).
ENV LLM_PROVIDER=groq \
    RATE_LIMIT_PER_MINUTE=6 \
    DAILY_QUERY_LIMIT=300 \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/health', timeout=4)"

# --proxy-headers: behind a host's reverse proxy, the client address used for
# rate limiting is the visitor's, not the proxy's.
CMD ["sh", "-c", "cd backend && exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
