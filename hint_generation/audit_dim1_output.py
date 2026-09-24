#!/usr/bin/env python3
"""Audit an L2/L3/L4 generation JSONL for count, order, and valid hints."""

import argparse
import json


REQUIRED_HINTS = ("L2_strategy", "L3_meta", "L4_classification")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def row_identity(row):
    sample_id = row.get("sample_id")
    if sample_id not in (None, ""):
        return ("sample_id", str(sample_id))
    problem = row.get("problem", row.get("Question"))
    return ("problem", problem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = load_jsonl(args.input)
    generated = load_jsonl(args.output)
    if len(generated) != len(source):
        raise ValueError(
            f"row-count mismatch: input={len(source)}, output={len(generated)}"
        )

    for index, (source_row, generated_row) in enumerate(zip(source, generated)):
        source_identity = row_identity(source_row)
        generated_identity = row_identity(generated_row)
        if generated_identity != source_identity:
            raise ValueError(
                f"row {index}: order/identity mismatch: "
                f"input={source_identity!r}, output={generated_identity!r}"
            )
        source_problem = source_row.get("problem", source_row.get("Question"))
        generated_problem = generated_row.get("problem", generated_row.get("Question"))
        source_solution = source_row.get("L1_full_solution", source_row.get("solution", source_row.get("COT_Reason")))
        generated_solution = generated_row.get("L1_full_solution", generated_row.get("solution", generated_row.get("COT_Reason")))
        if not isinstance(source_solution, str) or not source_solution.strip():
            raise ValueError(f"row {index}: missing reference solution")
        if source_problem != generated_problem or source_solution != generated_solution:
            raise ValueError(f"row {index}: problem/reference content mismatch")

        hints = generated_row.get("hints_abstraction")
        if not isinstance(hints, dict):
            raise ValueError(f"row {index}: missing hints_abstraction object")
        for key in REQUIRED_HINTS:
            value = hints.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"row {index}: missing or empty {key}")
            if value.lstrip().startswith("[ERROR"):
                raise ValueError(f"row {index}: error sentinel in {key}: {value}")
            if "<think>" in value.lower() or "</think>" in value.lower():
                raise ValueError(f"row {index}: thinking trace leaked into {key}")

    print(
        f"Audit passed: {len(generated):,} rows; exact input order; "
        f"all {', '.join(REQUIRED_HINTS)} values are non-empty and error-free."
    )


if __name__ == "__main__":
    main()
