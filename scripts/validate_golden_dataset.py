#!/usr/bin/env python3
"""Valide et nettoie les questions complexes du golden dataset."""

import json
from pathlib import Path

from eu_taxonomy_rag.evaluation.golden_dataset_validator import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    run_cleaning,
)


def main() -> None:
    cleaned, report = run_cleaning(
        input_path=DEFAULT_INPUT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
    )

    print(f"Cleaned dataset saved to: {DEFAULT_OUTPUT_PATH}")
    print(f"Total entries: {len(cleaned)}")
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
