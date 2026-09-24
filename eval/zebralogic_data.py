#!/usr/bin/env python3
"""Load and validate the locally downloaded ZebraLogic grid-mode test set."""

import re
from collections import Counter
from pathlib import Path


EXPECTED_NUM_PUZZLES = 1000

# These are the official ZebraLogic reporting buckets.
EASY_SIZES = frozenset({"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3"})
HARD_SIZES = frozenset(
    {
        "3*4", "3*5", "4*2", "3*6", "4*3", "4*4", "5*2", "6*2",
        "4*5", "4*6", "5*3", "5*4", "5*5", "5*6", "6*3", "6*4",
        "6*5", "6*6",
    }
)
SIZE_GROUPS = {
    "small": frozenset({"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2"}),
    "medium": frozenset({"3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2"}),
    "large": frozenset({"4*5", "5*3", "4*6", "5*4", "6*3"}),
    "xl": frozenset({"5*5", "6*4", "5*6", "6*5", "6*6"}),
}
ALL_SIZES = EASY_SIZES | HARD_SIZES


def difficulty_for_size(size):
    if size in EASY_SIZES:
        return "easy"
    if size in HARD_SIZES:
        return "hard"
    raise ValueError(f"Unknown ZebraLogic size: {size!r}")


def scale_group_for_size(size):
    matches = [name for name, sizes in SIZE_GROUPS.items() if size in sizes]
    if len(matches) != 1:
        raise ValueError(f"ZebraLogic size {size!r} belongs to {len(matches)} scale groups")
    return matches[0]


def _normalize_solution(solution, row_index, size):
    if not isinstance(solution, dict):
        raise ValueError(f"Row {row_index}: solution is not a struct/dict")
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        raise ValueError(f"Row {row_index}: solution must contain list-valued header and rows")
    header = [str(value).strip() for value in header]
    if not header or header[0] != "House" or any(not value for value in header):
        raise ValueError(f"Row {row_index}: malformed solution header: {header!r}")
    if len(set(header)) != len(header):
        raise ValueError(f"Row {row_index}: duplicate solution header fields: {header!r}")

    num_houses, num_attributes = map(int, size.split("*"))
    if len(rows) != num_houses:
        raise ValueError(
            f"Row {row_index}: size={size} but solution has {len(rows)} house rows"
        )
    if len(header) != num_attributes + 1:
        raise ValueError(
            f"Row {row_index}: size={size} but solution has {len(header) - 1} attributes"
        )

    normalized_rows = []
    for house_index, row in enumerate(rows, start=1):
        if not isinstance(row, list) or len(row) != len(header):
            raise ValueError(
                f"Row {row_index}, House {house_index}: expected {len(header)} cells, "
                f"found {len(row) if isinstance(row, list) else type(row).__name__}"
            )
        normalized = [str(value).strip() for value in row]
        if any(not value for value in normalized[1:]):
            raise ValueError(f"Row {row_index}, House {house_index}: empty gold cell")
        normalized_rows.append(normalized)

    hidden_answer_cells = [
        value
        for row in normalized_rows
        for value in row
        if value.strip("_").strip() == ""
    ]
    if hidden_answer_cells:
        raise ValueError(
            f"Row {row_index}: solution contains {len(hidden_answer_cells)} hidden-answer "
            "placeholders such as '___'. This is the answer-redacted "
            "allenai/ZebraLogicBench release and cannot be scored. Use the labeled "
            "WildEval/ZebraLogic grid_mode parquet instead."
        )

    # Every attribute in a logic grid is a one-to-one assignment across houses.
    for column_index, attribute in enumerate(header[1:], start=1):
        values = [row[column_index].casefold() for row in normalized_rows]
        if len(set(values)) != len(values):
            raise ValueError(f"Row {row_index}: duplicate gold values under {attribute!r}")

    table = {
        f"House {house_index}": {
            header[column_index]: row[column_index]
            for column_index in range(1, len(header))
        }
        for house_index, row in enumerate(normalized_rows, start=1)
    }
    return header, normalized_rows, table


def load_zebralogic(data_dir, check_expected_size=True):
    """Return validated grid-mode records from one or more local parquet shards."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("ZebraLogic loading requires pyarrow in the evaluation image") from exc

    grid_dir = Path(data_dir) / "grid_mode"
    files = sorted(grid_dir.glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ZebraLogic grid-mode test parquet found under {grid_dir}")

    required = {"id", "size", "puzzle", "solution"}
    records = []
    for parquet_file in files:
        table = pq.read_table(parquet_file)
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"{parquet_file} is missing columns: {sorted(missing)}")
        for source_row, example in enumerate(table.select(sorted(required)).to_pylist()):
            global_index = len(records)
            problem_id = str(example["id"]).strip()
            size = str(example["size"]).strip().lower().replace("x", "*")
            puzzle = str(example["puzzle"]).strip()
            if not problem_id:
                raise ValueError(f"Row {global_index}: empty id")
            if not re.fullmatch(r"[2-6]\*[2-6]", size) or size not in ALL_SIZES:
                raise ValueError(f"Row {global_index}: invalid/unsupported size {size!r}")
            if not puzzle:
                raise ValueError(f"Row {global_index}: empty puzzle")
            header, solution_rows, solution_table = _normalize_solution(
                example["solution"], global_index, size
            )
            records.append(
                {
                    "problem_id": problem_id,
                    "source_file": str(parquet_file),
                    "source_row": source_row,
                    "size": size,
                    "difficulty": difficulty_for_size(size),
                    "scale_group": scale_group_for_size(size),
                    "puzzle": puzzle,
                    "header": header,
                    "solution_rows": solution_rows,
                    "solution_table": solution_table,
                    "num_cells": (len(header) - 1) * len(solution_rows),
                }
            )

    ids = [record["problem_id"] for record in records]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate ZebraLogic ids: {duplicate_ids[:10]}")
    if check_expected_size and len(records) != EXPECTED_NUM_PUZZLES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_PUZZLES} ZebraLogic grid puzzles, found {len(records)}"
        )

    size_counts = Counter(record["size"] for record in records)
    difficulty_counts = Counter(record["difficulty"] for record in records)
    scale_counts = Counter(record["scale_group"] for record in records)
    print(f"Loaded ZebraLogic grid-mode test: {len(records)} puzzles from {len(files)} parquet file(s)")
    print(f"Size distribution: {dict(sorted(size_counts.items()))}")
    print(f"Difficulty distribution: {dict(sorted(difficulty_counts.items()))}")
    print(f"Scale distribution: {dict(sorted(scale_counts.items()))}")
    return records
