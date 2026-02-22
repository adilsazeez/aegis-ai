#!/usr/bin/env python3
"""
Download and cache the Prosocial Dialog dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/allenai/prosocial-dialog

Run from backend directory:
  python scripts/load_prosocial_dataset.py

Or from project root:
  python apps/backend/scripts/load_prosocial_dataset.py

First run downloads and caches the dataset (~117 MB). Later runs use the cache.
"""

import sys
from pathlib import Path

# Ensure backend root is on path so imports work if run from elsewhere
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

def main():
    print("Loading Prosocial Dialog dataset from Hugging Face...")
    print("(First run may download ~117 MB; subsequent runs use cache.)\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. From apps/backend run:")
        print("  pip install datasets")
        sys.exit(1)

    # Load the dataset; uses default config and all splits (train, validation, test)
    # Cached under ~/.cache/huggingface/datasets/ by default
    dataset = load_dataset("allenai/prosocial-dialog")

    print("Dataset loaded successfully.\n")
    print("Splits and row counts:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data):,} rows")
    print("\nColumns:", list(dataset["train"].column_names))
    print("\nSample row (train, first):")
    first = dataset["train"][0]
    for key, value in first.items():
        val_str = str(value)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        print(f"  {key}: {val_str}")

    print("\nDone. Dataset is cached for use by the RAG pipeline.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
