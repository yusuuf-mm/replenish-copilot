FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HUB_OFFLINE=0

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Bake the embedding model into the image so cold starts need no HF access.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/
COPY README.md architecture.md ./

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
