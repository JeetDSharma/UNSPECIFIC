#!/usr/bin/env python3
"""
Pairwise cosine similarity between two text columns from CSVs joined on a key column
(e.g. direct_content vs base_content aligned by instruction_number).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from cs4.utils.embedding_utils import cosine_similarity_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity (sentence-transformers) between aligned text columns"
    )
    parser.add_argument(
        "--reference-path",
        required=True,
        help="CSV containing the reference column (e.g. base_generated with base_content)",
    )
    parser.add_argument(
        "--reference-column",
        default="base_content",
        help="Reference text column (default: base_content)",
    )
    parser.add_argument(
        "--response-path",
        required=True,
        help="CSV containing the response column (e.g. direct_generated with direct_content)",
    )
    parser.add_argument(
        "--response-column",
        default="direct_content",
        help="Response text column (default: direct_content)",
    )
    parser.add_argument(
        "--key",
        default="instruction_number",
        help="Join key present in both CSVs (default: instruction_number)",
    )
    parser.add_argument(
        "--model",
        default="all-mpnet-base-v2",
        help="sentence-transformers model id",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_path, encoding="utf-8")
    resp_df = pd.read_csv(args.response_path, encoding="utf-8")

    for name, df, col in (
        ("reference", ref_df, args.reference_column),
        ("response", resp_df, args.response_column),
    ):
        if col not in df.columns:
            print(f"Error: {name} CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)
    if args.key not in ref_df.columns or args.key not in resp_df.columns:
        print(f"Error: join key '{args.key}' missing from one or both CSVs", file=sys.stderr)
        sys.exit(1)

    for name, df in ("reference", ref_df), ("response", resp_df):
        dup = df[args.key].duplicated().any()
        if dup:
            print(
                f"Error: duplicate {args.key} values in {name} CSV",
                file=sys.stderr,
            )
            sys.exit(1)

    merged = pd.merge(
        ref_df[[args.key, args.reference_column]],
        resp_df[[args.key, args.response_column]],
        on=args.key,
        how="inner",
    )
    merged = merged.sort_values(args.key).reset_index(drop=True)
    if len(merged) == 0:
        print("Error: inner join produced no rows", file=sys.stderr)
        sys.exit(1)

    texts_a = merged[args.reference_column].astype(str).tolist()
    texts_b = merged[args.response_column].astype(str).tolist()
    sims = cosine_similarity_pairs(texts_a, texts_b, model_name=args.model)

    out_rows = []
    for idx, (_, row) in enumerate(merged.iterrows()):
        key_val = row[args.key]
        a = row[args.reference_column]
        b = row[args.response_column]
        ref_len = len((a or "").strip()) if pd.notna(a) else 0
        resp_len = len((b or "").strip()) if pd.notna(b) else 0
        cos = sims[idx]
        out_rows.append(
            {
                args.key: key_val,
                "cosine_similarity": float(cos) if not np.isnan(cos) else np.nan,
                "reference_length": ref_len,
                "response_length": resp_len,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    valid = sims[~np.isnan(sims)]
    if len(valid) > 0:
        print(
            f"Wrote {len(out_rows)} rows to {out_path} | "
            f"cosine mean={float(np.mean(valid)):.4f} "
            f"min={float(np.min(valid)):.4f} max={float(np.max(valid)):.4f}"
        )
    else:
        print(f"Wrote {len(out_rows)} rows to {out_path} (all pairs empty)")


if __name__ == "__main__":
    main()
