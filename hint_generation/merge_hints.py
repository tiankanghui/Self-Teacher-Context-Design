"""
Merge generated abstraction-level hints into the original training parquet.

Usage:
  python merge_hints.py \
    --hints /path/to/train_dim1.jsonl \
    --parquet "/path/to/original_data/train*.parquet" \
    --output /path/to/output/ \
    --level L2_strategy
"""
import argparse
import json
import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hints", required=True, help="Hints JSONL file")
    parser.add_argument("--parquet", required=True, help="Original parquet glob pattern")
    parser.add_argument("--output", required=True, help="Output directory for merged parquet")
    parser.add_argument("--level", default="L2_strategy",
                        choices=["L1_concrete", "L2_strategy", "L3_meta", "L4_classification"],
                        help="Hint level to use as teacher solution (default: L2_strategy)")
    args = parser.parse_args()

    # Load hints
    hints_data = []
    with open(args.hints, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                hints_data.append(json.loads(line))
    print(f"Loaded {len(hints_data)} hints from {args.hints}")

    # Load original parquet
    parquet_files = sorted(glob.glob(args.parquet))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(parquet_files)} parquet files")

    assert len(df) == len(hints_data), \
        f"Row count mismatch: parquet={len(df)}, hints={len(hints_data)}"

    # Replace solution with selected hint level (or original solution for L1)
    if args.level == "L1_concrete":
        print("Using original solution (L1), no replacement needed")
    else:
        new_solutions = []
        missing = 0
        for h in hints_data:
            ab = h.get("hints_abstraction", {})
            hint = ab.get(args.level, "")
            if not hint or hint.startswith("[ERROR"):
                # Fallback to original solution if hint generation failed
                missing += 1
                hint = "[HINT_MISSING]"
            new_solutions.append(hint)
        df["solution"] = new_solutions
        print(f"Replaced solution with {args.level} hints ({missing} fallback)")

    # Save
    import os
    os.makedirs(args.output, exist_ok=True)
    out_path = f"{args.output}/train_hint_{args.level}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
