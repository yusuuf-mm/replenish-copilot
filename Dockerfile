FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HUB_OFFLINE=0 \
    HF_HOME=/app/.hf-cache \
    UV_HTTP_TIMEOUT=120

COPY pyproject.toml uv.lock ./
# uv cache mount: wheel downloads shared across builds/projects, so a failed
# or repeated build reuses everything instead of re-downloading.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

# Bake the embedding model into a real image layer (NOT a cache mount: cache
# mounts are discarded after the build). Runtime sets HF_HUB_OFFLINE=1 and
# reuses /app/.hf-cache with no network.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/
COPY README.md architecture.md ./

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
