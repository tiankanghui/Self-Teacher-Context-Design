#!/usr/bin/env python3
"""Merge GPQA-Diamond data-parallel shards and compute anti-forgetting deltas."""

import argparse
import glob
import json
from pathlib import Path


def load_run(name, prefix, expected_val_n):
    files = sorted(glob.glob(f"{prefix}_shard*.json"))
    if not files:
        print(f"[WARN] No shards for {name}: {prefix}_shard*.json")
        return None

    aggregate = {
        "num_shards": len(files),
        "num_problems": 0,
        "total_solutions": 0,
        "correct": 0,
        "pass_at_n": 0,
        "majority_vote_at_n": 0,
        "extraction_failures": 0,
    }
    seen_problem_ids = set()
    for file_name in files:
        with open(file_name, encoding="utf-8") as f:
            shard = json.load(f)
        if shard.get("dataset") != "gpqa-diamond":
            raise ValueError(f"Unexpected dataset in {file_name}: {shard.get('dataset')}")
        if shard.get("val_n") != expected_val_n:
            raise ValueError(f"val_n mismatch in {file_name}")
        if len(shard.get("results", [])) != shard.get("num_problems"):
            raise ValueError(f"Incomplete result list in {file_name}")

        problem_ids = {row["problem_id"] for row in shard["results"]}
        overlap = seen_problem_ids & problem_ids
        if overlap:
            raise ValueError(f"Duplicate problem IDs across shards for {name}: {sorted(overlap)}")
        seen_problem_ids.update(problem_ids)

        aggregate["num_problems"] += shard["num_problems"]
        aggregate["total_solutions"] += shard["total_solutions"]
        aggregate["correct"] += shard["average_at_n"]
        aggregate["pass_at_n"] += shard["pass_at_n"]
        aggregate["majority_vote_at_n"] += shard["majority_vote_at_n"]
        aggregate["extraction_failures"] += shard["extraction_failures"]

    if aggregate["num_problems"] != 198:
        raise ValueError(
            f"Expected 198 unique GPQA-Diamond problems for {name}, "
            f"found {aggregate['num_problems']}"
        )
    total = aggregate["total_solutions"]
    problems = aggregate["num_problems"]
    aggregate["average_at_n_pct"] = aggregate["correct"] / total * 100
    aggregate["pass_at_n_pct"] = aggregate["pass_at_n"] / problems * 100
    aggregate["majority_vote_at_n_pct"] = (
        aggregate["majority_vote_at_n"] / problems * 100
    )
    aggregate["extraction_failure_rate"] = (
        aggregate["extraction_failures"] / total * 100
    )
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PREFIX")
    parser.add_argument("--val_n", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Invalid --run value: {spec}")
        name, prefix = spec.split("=", 1)
        result = load_run(name, prefix, args.val_n)
        if result is not None:
            results[name] = result
    if not results:
        raise RuntimeError("No completed GPQA-Diamond runs found")

    base = results.get("base")
    l1 = next((value for name, value in results.items() if name.startswith("L1-step")), None)
    for metrics in results.values():
        score = metrics["average_at_n_pct"]
        if base is not None:
            base_score = base["average_at_n_pct"]
            metrics["ood_change_vs_base"] = score - base_score
            metrics["forgetting_vs_base"] = base_score - score
        if l1 is not None:
            metrics["gain_over_l1"] = score - l1["average_at_n_pct"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGPQA-Diamond summary (Avg@{args.val_n})")
    print(
        f"{'Run':<14} {'Avg':>8} {'Pass':>8} {'Majority':>10} "
        f"{'ExtractFail':>12} {'vs Base':>9} {'vs L1':>9}"
    )
    for name, metrics in results.items():
        versus_base = metrics.get("ood_change_vs_base")
        versus_l1 = metrics.get("gain_over_l1")
        print(
            f"{name:<14} {metrics['average_at_n_pct']:>8.2f} "
            f"{metrics['pass_at_n_pct']:>8.2f} "
            f"{metrics['majority_vote_at_n_pct']:>10.2f} "
            f"{metrics['extraction_failure_rate']:>12.2f} "
            f"{('n/a' if versus_base is None else f'{versus_base:.2f}'):>9} "
            f"{('n/a' if versus_l1 is None else f'{versus_l1:.2f}'):>9}"
        )
    print(f"\nDetailed summary: {output}")


if __name__ == "__main__":
    main()
