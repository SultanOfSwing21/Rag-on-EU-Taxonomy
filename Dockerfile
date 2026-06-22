# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Layer 1 — dependency install (rebuilds only when pyproject.toml / README change)
COPY pyproject.toml README.md ./
RUN mkdir -p src/eu_taxonomy_rag \
    && printf '%s\n' '"""EU Taxonomy RAG package."""' > src/eu_taxonomy_rag/__init__.py \
    && printf '%s\n' \
        'def main() -> None:' \
        '    """Placeholder until application source is copied."""' \
        '    raise SystemExit("Application source not mounted")' \
        > src/eu_taxonomy_rag/cli.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -e ".[ui]"

# Layer 2 — application source (code changes do not reinstall Python dependencies)
COPY src ./src
COPY app ./app
COPY data ./data
COPY scripts ./scripts

ENV EU_TAXONOMY_PROJECT_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["eu-taxonomy-rag"]
