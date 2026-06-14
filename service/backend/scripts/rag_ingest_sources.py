from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.backend.app.rag.ingest import ingest_rag_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RAG rulebook, SOP, and dataset case summaries.")
    parser.add_argument("--dataset-limit", type=int, default=None, help="Limit dataset case summaries for local smoke runs.")
    args = parser.parse_args()
    chunk_count = ingest_rag_sources(dataset_limit=args.dataset_limit)
    print(f"RAG ingestion completed: {chunk_count} chunks.")


if __name__ == "__main__":
    main()
