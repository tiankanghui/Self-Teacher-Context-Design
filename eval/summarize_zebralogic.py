#!/usr/bin/env python3
"""Merge ZebraLogic shards and report full-grid, cell, and difficulty metrics."""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from zebralogic_data import EXPECTED_NUM_PUZZLES


def percentage(numerator, denominator):
    return numerator / denominator * 100 if denominator else None


def load_run(
    name,
    prefix,
    expected_val_n,
    expected_max_new_tokens=None,
    expected_max_model_len=None,
):
    files = sorted(glob.glob(f"{prefix}_shard*.json"))
    if not files:
        print(f"[WARN] No shards for {name}: {prefix}_shard*.json")
        return None

    rows = []
    seen_ids = set()
    for file_name in files:
        with open(file_name, encoding="utf-8") as f:
            shard = json.load(f)
        if shard.get("dataset") != "zebralogic-grid" or shard.get("split") != "test":
            raise ValueError(f"Dataset/split mismatch in {file_name}")
        if shard.get("val_n") != expected_val_n:
            raise ValueError(f"val_n mismatch in {file_name}")
        if (
            expected_max_new_tokens is not None
            and shard.get("max_new_tokens") != expected_max_new_tokens
        ):
            raise ValueError(f"max_new_tokens mismatch in {file_name}")
        if (
            expected_max_model_len is not None
            and shard.get("max_model_len") != expected_max_model_len
        ):
            raise ValueError(f"max_model_len mismatch in {file_name}")
        shard_rows = shard.get("results", [])
        if len(shard_rows) != shard.get("num_problems"):
            raise ValueError(f"Incomplete result list in {file_name}")
        for row in shard_rows:
            problem_id = row["problem_id"]
            if problem_id in seen_ids:
                raise ValueError(f"Duplicate problem ID for {name}: {problem_id}")
            seen_ids.add(problem_id)
            rows.append(row)

    if len(rows) != EXPECTED_NUM_PUZZLES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_PUZZLES} unique ZebraLogic puzzles for {name}, "
            f"found {len(rows)}"
        )

    def aggregate(key_name=None):
        grouped = defaultdict(list)
        if key_name is None:
            grouped["all"] = rows
        else:
            for row in rows:
                grouped[row[key_name]].append(row)
        result = {}
        for key, members in grouped.items():
            solved = sum(int(row["solved"]) for row in members)
            correct_cells = sum(row["correct_cells"] for row in members)
            total_cells = sum(row["total_cells"] for row in members)
            parsed = sum(int(row["parsed"]) for row in members)
            result[key] = {
                "puzzles": len(members),
                "solved": solved,
                "puzzle_accuracy_pct": percentage(solved, len(members)),
                "correct_cells": correct_cells,
                "total_cells": total_cells,
                "cell_accuracy_pct": percentage(correct_cells, total_cells),
                "no_answer": len(members) - parsed,
                "no_answer_rate": percentage(len(members) - parsed, len(members)),
            }
        return result

    overall = aggregate()["all"]
    length_cutoffs = sum(row.get("finish_reason") == "length" for row in rows)
    return {
        "num_shards": len(files),
        "split": "test",
        "val_n": expected_val_n,
        "num_puzzles": len(rows),
        **overall,
        "length_cutoffs": length_cutoffs,
        "length_cutoff_rate": percentage(length_cutoffs, len(rows)),
        "by_size": aggregate("size"),
        "by_difficulty": aggregate("difficulty"),
        "by_scale": aggregate("scale_group"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PREFIX")
    parser.add_argument("--val_n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--max_model_len", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.val_n != 1:
        raise ValueError("ZebraLogic summary requires --val_n 1")
    results = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Invalid --run value: {spec}")
        name, prefix = spec.split("=", 1)
        result = load_run(
            name,
            prefix,
            args.val_n,
            args.max_new_tokens,
            args.max_model_len,
        )
        if result is not None:
            results[name] = result
    if not results:
        raise RuntimeError("No completed ZebraLogic runs found")

    base = results.get("base")
    l1 = next((value for name, value in results.items() if name.startswith("L1-step")), None)
    for metrics in results.values():
        if base is not None:
            metrics["puzzle_change_vs_base"] = (
                metrics["puzzle_accuracy_pct"] - base["puzzle_accuracy_pct"]
            )
            metrics["cell_change_vs_base"] = (
                metrics["cell_accuracy_pct"] - base["cell_accuracy_pct"]
            )
        if l1 is not None:
            metrics["puzzle_gain_over_l1"] = (
                metrics["puzzle_accuracy_pct"] - l1["puzzle_accuracy_pct"]
            )
            metrics["cell_gain_over_l1"] = (
                metrics["cell_accuracy_pct"] - l1["cell_accuracy_pct"]
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nZebraLogic grid-mode summary (single-sample full-grid evaluation)")
    print(
        f"{'Run':<14} {'Puzzle':>8} {'Cell':>8} {'Hard-P':>8} {'XL-P':>8} "
        f"{'NoAns':>8} {'LenCut':>8} {'vs Base':>9} {'vs L1':>9}"
    )
    for name, metrics in results.items():
        hard = metrics["by_difficulty"].get("hard", {}).get("puzzle_accuracy_pct")
        xl = metrics["by_scale"].get("xl", {}).get("puzzle_accuracy_pct")
        versus_base = metrics.get("puzzle_change_vs_base")
        versus_l1 = metrics.get("puzzle_gain_over_l1")
        print(
            f"{name:<14} {metrics['puzzle_accuracy_pct']:>8.2f} "
            f"{metrics['cell_accuracy_pct']:>8.2f} "
            f"{('n/a' if hard is None else f'{hard:.2f}'):>8} "
            f"{('n/a' if xl is None else f'{xl:.2f}'):>8} "
            f"{metrics['no_answer_rate']:>8.2f} "
            f"{metrics['length_cutoff_rate']:>8.2f} "
            f"{('n/a' if versus_base is None else f'{versus_base:.2f}'):>9} "
            f"{('n/a' if versus_l1 is None else f'{versus_l1:.2f}'):>9}"
        )
    print(f"\nDetailed size/difficulty summary: {output_path}")


if __name__ == "__main__":
    main()
