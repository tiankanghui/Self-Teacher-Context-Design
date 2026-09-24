#!/usr/bin/env python3
"""Format official GPQA-Diamond using the exact option order used in the paper."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(str(text).strip().encode()).hexdigest()


def format_rows(raw, manifest):
    by_question = {digest(row["Question"]): row for row in raw}
    if len(by_question) != len(raw) or len(raw) != len(manifest):
        raise ValueError("Unexpected GPQA row count or duplicate questions")
    result = []
    for item in manifest:
        if item["question_sha256"] not in by_question:
            raise ValueError("GPQA question differs from the frozen question fingerprint")
        row = by_question[item["question_sha256"]]
        options = [str(row[k]).strip() for k in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
        by_hash = {digest(value): value for value in options}
        if Counter(digest(v) for v in options) != Counter(item["option_sha256"]):
            raise ValueError("GPQA answer choices differ from the frozen option fingerprints")
        ordered = [prefix + by_hash[h] + suffix for h, (prefix, suffix) in
                   zip(item["option_sha256"], item["option_whitespace"])]
        if ordered["ABCD".index(item["answer"])].strip() != options[0]:
            raise ValueError("GPQA correct answer differs from the frozen answer key")
        prefix, suffix = item["question_whitespace"]
        result.append({"question": prefix + str(row["Question"]).strip() + suffix + "\n\n" +
                       "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", ordered)),
                       "answer": item["answer"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Official raw GPQA-Diamond Parquet or CSV")
    source.add_argument("--download", action="store_true", help="Use authorized Hugging Face access")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval/gpqa_diamond/gpqa_diamond.parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Output exists; pass --overwrite to replace it")
    import pyarrow as pa
    import pyarrow.parquet as pq
    if args.download:
        from datasets import load_dataset
        raw = list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
    elif args.input.suffix == ".csv":
        with args.input.open(newline="") as handle:
            raw = list(csv.DictReader(handle))
    else:
        raw = pq.read_table(args.input).to_pylist()
    manifest = json.loads((ROOT / "manifests/gpqa_option_order.json").read_text())
    rows = format_rows(raw, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"Formatted {len(rows)} GPQA-Diamond problems with the frozen option order")


if __name__ == "__main__":
    main()
