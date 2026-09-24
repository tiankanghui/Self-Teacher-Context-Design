#!/usr/bin/env python3
"""Merge AutoLogi data-parallel shards and compute anti-forgetting deltas."""

import argparse
import glob
import json
from pathlib import Path


def load_run(name, prefix, expected_split, expected_val_n):
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
        "extraction_failures": 0,
        "verifier_errors": 0,
        "verifier_timeouts": 0,
        "by_constraint_count": {},
    }
    for file_name in files:
        with open(file_name, encoding="utf-8") as f:
            shard = json.load(f)
        if shard.get("dataset") != "autologi":
            raise ValueError(f"Unexpected dataset in {file_name}: {shard.get('dataset')}")
        if shard.get("autologi_split") != expected_split or shard.get("val_n") != expected_val_n:
            raise ValueError(f"Configuration mismatch in {file_name}")

        aggregate["num_problems"] += shard["num_problems"]
        aggregate["total_solutions"] += shard["total_solutions"]
        aggregate["correct"] += shard["average_at_n"]
        aggregate["pass_at_n"] += shard["pass_at_n"]
        aggregate["extraction_failures"] += shard["extraction_failures"]
        aggregate["verifier_errors"] += shard["verifier_errors"]
        aggregate["verifier_timeouts"] += shard["verifier_timeouts"]
        for count, metrics in shard.get("by_constraint_count", {}).items():
            bucket = aggregate["by_constraint_count"].setdefault(
                count, {"num_problems": 0, "correct": 0, "total": 0}
            )
            for key in ("num_problems", "correct", "total"):
                bucket[key] += metrics[key]

    total = aggregate["total_solutions"]
    problems = aggregate["num_problems"]
    aggregate["average_at_n_pct"] = aggregate["correct"] / total * 100
    aggregate["pass_at_n_pct"] = aggregate["pass_at_n"] / problems * 100
    aggregate["extraction_failure_rate"] = aggregate["extraction_failures"] / total * 100
    aggregate["verifier_error_rate"] = aggregate["verifier_errors"] / total * 100
    aggregate["verifier_timeout_rate"] = aggregate["verifier_timeouts"] / total * 100
    for bucket in aggregate["by_constraint_count"].values():
        bucket["accuracy_pct"] = bucket["correct"] / bucket["total"] * 100
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PREFIX")
    parser.add_argument("--split", choices=["en", "cn", "both"], default="en")
    parser.add_argument("--val_n", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Invalid --run value: {spec}")
        name, prefix = spec.split("=", 1)
        result = load_run(name, prefix, args.split, args.val_n)
        if result is not None:
            results[name] = result
    if not results:
        raise RuntimeError("No completed AutoLogi runs found")

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

    print(f"\nAutoLogi-{args.split.upper()} summary (val_n={args.val_n})")
    print(f"{'Run':<14} {'Accuracy':>9} {'ExtractFail':>12} {'VerifyErr':>10} {'vs Base':>9} {'vs L1':>9}")
    for name, metrics in results.items():
        versus_base = metrics.get("ood_change_vs_base")
        versus_l1 = metrics.get("gain_over_l1")
        print(
            f"{name:<14} {metrics['average_at_n_pct']:>9.2f} "
            f"{metrics['extraction_failure_rate']:>12.2f} "
            f"{metrics['verifier_error_rate']:>10.2f} "
            f"{('n/a' if versus_base is None else f'{versus_base:.2f}'):>9} "
            f"{('n/a' if versus_l1 is None else f'{versus_l1:.2f}'):>9}"
        )
    print(f"\nDetailed summary: {output}")


if __name__ == "__main__":
    main()
