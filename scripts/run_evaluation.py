#!/usr/bin/env python3
"""Exécute le benchmark retrieval (Recall@K + MRR) et exporte les résultats."""

import argparse
from pathlib import Path

from eu_taxonomy_rag.evaluation.natural_dataset import NATURAL_DATASET_748_PATH
from eu_taxonomy_rag.evaluation.runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_DIR,
    ALL_RETRIEVAL_METHODS,
    default_output_path,
    format_metrics_summary,
    run_retrieval_evaluation,
)
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod


def _parse_methods(raw: str) -> list[RetrievalMethod]:
    if raw.strip().lower() == "all":
        return list(ALL_RETRIEVAL_METHODS)

    methods: list[RetrievalMethod] = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        methods.append(RetrievalMethod(name))
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Évalue les méthodes de retrieval sur un dataset JSONL.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Chemin du dataset JSONL (golden cleaned par défaut).",
    )
    parser.add_argument(
        "--natural-748",
        action="store_true",
        help=f"Raccourci pour --dataset {NATURAL_DATASET_748_PATH}",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Méthodes séparées par des virgules, ou 'all'.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Nombre de chunks récupérés par requête (défaut: 5).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Taille du pool candidat pour l'hybride (défaut: 20).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(".cache/index"),
        help="Répertoire des index FAISS/BM25.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier JSON de sortie (défaut: data/evaluation/results/...).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite le nombre de questions (debug / smoke test).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = NATURAL_DATASET_748_PATH if args.natural_748 else args.dataset
    methods = _parse_methods(args.methods)
    output_path = args.output or default_output_path(dataset_path, results_dir=DEFAULT_RESULTS_DIR)

    run_result = run_retrieval_evaluation(
        dataset_path=dataset_path,
        methods=methods,
        k=args.k,
        candidate_k=args.candidate_k,
        index_dir=args.index_dir,
        limit=args.limit,
    )
    saved_path = run_result.save(output_path)

    print(format_metrics_summary(run_result))
    print()
    print(f"Results saved to: {saved_path}")


if __name__ == "__main__":
    main()
