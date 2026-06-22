#!/usr/bin/env python3
"""Génère le dataset de requêtes utilisateur naturelles via LLM."""

import json

from eu_taxonomy_rag.evaluation.natural_dataset import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SOURCE_PATH,
    run_natural_dataset_generation,
)


def main() -> None:
    dataset, stats = run_natural_dataset_generation(
        source_path=DEFAULT_SOURCE_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        n_simple=200,
        n_complex=100,
    )

    print(f"Natural dataset saved to: {DEFAULT_OUTPUT_PATH}")
    print(f"Total queries: {len(dataset)}")
    print(json.dumps(stats.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
