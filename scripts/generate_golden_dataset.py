#!/usr/bin/env python3
"""Génère le golden dataset de retrieval et affiche les statistiques."""

import json
from pathlib import Path

from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.evaluation.golden_dataset import (
    DEFAULT_OUTPUT_PATH,
    GenerationConfig,
    generate_golden_dataset,
    save_golden_dataset,
)


def main() -> None:
    chunks = load_or_build_chunks()
    config = GenerationConfig()
    dataset, stats = generate_golden_dataset(chunks, config)
    output_path = save_golden_dataset(dataset, DEFAULT_OUTPUT_PATH)

    print(f"Golden dataset saved to: {output_path}")
    print(json.dumps(stats.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
