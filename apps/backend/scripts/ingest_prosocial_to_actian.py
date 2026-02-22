#!/usr/bin/env python3
"""
Ingest Prosocial Dialog (train split) into Actian VectorAI DB.
Embeds the "context" field and stores payload with context, response, rots, safety_*, etc.
Requires: Actian container running (e.g. localhost:50051), cortex package installed.

Run from backend directory:
  python scripts/ingest_prosocial_to_actian.py

Or from project root:
  python apps/backend/scripts/ingest_prosocial_to_actian.py
"""

import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

import rag_config
from embedding import encode
from actian_client import get_sync_client, is_actian_available

BATCH_SIZE = 500


def main():
    if not is_actian_available():
        print("ERROR: Actian Cortex client not installed.")
        print("Install the wheel from https://github.com/hackmamba-io/actian-vectorAI-db-beta")
        return 1

    print("Loading Prosocial Dialog (train split)...")
    from datasets import load_dataset
    ds = load_dataset("allenai/prosocial-dialog", split="train")
    texts = [row.get("context") or "" for row in ds]
    n = len(texts)
    print(f"Loaded {n:,} rows. Embedding 'context' with {rag_config.EMBEDDING_MODEL}...")

    client, DM = get_sync_client()
    with client:
        coll = rag_config.PROSOCIAL_COLLECTION
        dim = rag_config.EMBEDDING_DIMENSION
        if client.has_collection(coll):
            print(f"Collection '{coll}' already exists. Dropping to re-ingest.")
            client.delete_collection(coll)
        client.create_collection(
            name=coll,
            dimension=dim,
            distance_metric=DM.COSINE,
            hnsw_ef_search=100,
        )
        print(f"Created collection '{coll}' (dim={dim}).")

        inserted = 0
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch_texts = texts[start:end]
            vectors = encode(batch_texts, batch_size=64)
            ids = [start + i for i in range(len(batch_texts))]  # Actian expects int IDs (u64)
            payloads = []
            for i, idx in enumerate(range(start, end)):
                row = ds[idx]
                payloads.append({
                    "context": row.get("context") or "",
                    "response": row.get("response") or "",
                    "rots": str(row.get("rots", "")),
                    "safety_label": str(row.get("safety_label", "")),
                    "safety_annotations": str(row.get("safety_annotations", "")),
                    "safety_annotation_reasons": str(row.get("safety_annotation_reasons", "")),
                    "source": str(row.get("source", "")),
                    "dialogue_id": str(row.get("dialogue_id", "")),
                    "response_id": str(row.get("response_id", "")),
                })
            client.batch_upsert(coll, ids, vectors, payloads)
            inserted += len(ids)
            print(f"  Upserted {inserted:,} / {n:,}")
        print(f"Done. Ingested {inserted:,} vectors into '{coll}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
