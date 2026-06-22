# syntax=docker/dockerfile:1
#
# Image applicative EU Taxonomy RAG (Streamlit + pipeline RAG).
# Les jeux d'évaluation versionnés (golden, natural) sont copiés dans l'image via COPY data/.
# La documentation in-app (onglet Documentation, page d'accueil) via COPY docs/documentation/.
#
# Persistance (non incluse dans l'image — montée par docker-compose.yml) :
#   /app/.cache                      chunks.jsonl, index/, generation_eval.db (SQLite)
#   /app/data/evaluation/results     exports JSON des benchmarks retrieval
#   /root/.cache/huggingface         modèles téléchargés (volume nommé hf-cache)
#
# Démarrage : docker compose up --build  →  http://localhost:8501
#
# PyTorch CPU-only (index pytorch.org) : pas de paquets NVIDIA CUDA dans l'image Docker.
# L'installation locale via pyproject.toml reste inchangée (pip / GPU selon l'environnement hôte).

FROM python:3.11-slim

ARG TORCH_VERSION=2.2.2

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
    && pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[ui]"

# Layer 2 — application source (code changes do not reinstall Python dependencies)
COPY src ./src
COPY app ./app
# Jeux d'évaluation versionnés (golden, natural) — les résultats de benchmark
# sont écrits dans data/evaluation/results/ et doivent être montés en volume (voir docker-compose.yml).
COPY data ./data
# Markdown servi par l'onglet Documentation et la page d'accueil (home.md)
COPY docs/documentation ./docs/documentation
COPY scripts ./scripts

ENV EU_TAXONOMY_PROJECT_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["eu-taxonomy-rag"]
