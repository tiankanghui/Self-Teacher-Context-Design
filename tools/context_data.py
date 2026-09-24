#!/usr/bin/env python3
"""Validate the release, merge regenerated hints, or export per-level Parquet.

All commands preserve problem order and reference strings. Generated files
belong in outputs/; the repository needs only the original aligned JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

LEVEL_FIELDS = {
    "L1": "L1_full_solution", "L2": "L2_strategy", "L3": "L3_framing",
    "L4": "L4_category", "L5": "L5_answer_only",
}
GENERATED_FIELDS = {"L2_strategy": "L2_strategy", "L3_framing": "L3_meta", "L4_category": "L4_classification"}


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"{path.name}:{index + 1}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{index + 1}: expected an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: empty dataset")
    return rows


def require_text(value: object, *, field: str, index: int, hint: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {index}: missing or empty {field}")
    if value.lstrip().startswith("[ERROR") or value.strip() == "[HINT_MISSING]":
        raise ValueError(f"Row {index}: error marker in {field}")
    if hint and any(tag in value.lower() for tag in ("<think>", "</think>")):
        raise ValueError(f"Row {index}: thinking trace in {field}")
    return value


def validate_rows(rows: list[dict]) -> None:
    ids = set()
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if type(sample_id) is not int or sample_id < 0 or sample_id in ids:
            raise ValueError(f"Row {index}: invalid or duplicate sample_id")
        ids.add(sample_id)
        require_text(row.get("problem"), field="problem", index=index)
        if not isinstance(row.get("source"), str):
            raise ValueError(f"Row {index}: source must be a string")
        for level, field in LEVEL_FIELDS.items():
            require_text(row.get(field), field=field, index=index, hint=level in ("L2", "L3", "L4"))


def merge_rows(reference: list[dict], generated: list[dict]) -> list[dict]:
    validate_rows(reference)
    if len(reference) != len(generated):
        raise ValueError("Reference and generated row counts differ")
    merged = []
    for index, (ref, gen) in enumerate(zip(reference, generated)):
        problem = gen.get("problem", gen.get("Question"))
        if problem != ref["problem"]:
            raise ValueError(f"Row {index}: problem/order mismatch")
        if "sample_id" in gen and gen["sample_id"] != ref["sample_id"]:
            raise ValueError(f"Row {index}: sample_id mismatch")
        solution = gen.get("L1_full_solution", gen.get("solution", gen.get("COT_Reason")))
        if solution != ref["L1_full_solution"]:
            raise ValueError(f"Row {index}: reference solution mismatch")
        hints = gen.get("hints_abstraction")
        if not isinstance(hints, dict):
            raise ValueError(f"Row {index}: missing hints_abstraction")
        row = dict(ref)
        for target, key in GENERATED_FIELDS.items():
            row[target] = require_text(hints.get(key), field=key, index=index, hint=True)
        merged.append(row)
    return merged


def write_jsonl(path: Path, rows: list[dict], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--reference", type=Path, required=True)
    merge.add_argument("--generated", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    export = commands.add_parser("parquet")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "merge":
        if args.output.resolve() in (args.reference.resolve(), args.generated.resolve()):
            raise ValueError("Write merged data to a separate file")
        rows = merge_rows(read_rows(args.reference), read_rows(args.generated))
        write_jsonl(args.output, rows, overwrite=args.overwrite)
        print(f"Merged {len(rows):,} rows into {args.output}")
        return
    rows = read_rows(args.input)
    validate_rows(rows)
    if args.command == "validate":
        print(f"Validated {len(rows):,} aligned L1--L5 rows")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = {level: args.output_dir / level / "train.parquet" for level in LEVEL_FIELDS}
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("An output Parquet already exists; use --overwrite")
    for level, field in LEVEL_FIELDS.items():
        data = [{"sample_id": row["sample_id"], "source": row["source"],
                 "problem": row["problem"], "solution": row[field]} for row in rows]
        paths[level].parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(data), paths[level])
    print(f"Exported {len(rows):,} rows for each level into {args.output_dir}")


if __name__ == "__main__":
    main()
