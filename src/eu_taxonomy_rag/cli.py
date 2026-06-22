"""Command-line entry point to bootstrap the app and launch Streamlit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eu_taxonomy_rag.paths import DEFAULT_FAQ_PATH, PROJECT_ROOT


def bootstrap_application(*, force_rebuild: bool = False) -> int:
    """Build FAQ chunks and initialize evaluation storage (indexes are built in the UI)."""
    from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
    from eu_taxonomy_rag.storage.evaluation_store import init_evaluation_db

    if not DEFAULT_FAQ_PATH.exists():
        raise FileNotFoundError(
            f"FAQ dataset not found at `{DEFAULT_FAQ_PATH}`. "
            "Make sure you cloned the full repository."
        )

    init_evaluation_db()
    chunks = load_or_build_chunks(force_rebuild=force_rebuild)
    print(f"Chunks ready: {len(chunks)}")
    print("Indexes: build them from the Benchmark page in Streamlit.")
    return len(chunks)


def launch_streamlit() -> int:
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at `{app_path}`.")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap EU Taxonomy RAG and launch the Streamlit dashboard.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Build FAQ chunks without starting Streamlit.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild FAQ chunks from scratch.",
    )
    args = parser.parse_args(argv)

    print("EU Taxonomy RAG — preparing FAQ chunks and evaluation storage…")
    bootstrap_application(force_rebuild=args.force_rebuild)

    if args.bootstrap_only:
        print("Bootstrap complete.")
        return

    print("Starting Streamlit…")
    raise SystemExit(launch_streamlit())


if __name__ == "__main__":
    main()
