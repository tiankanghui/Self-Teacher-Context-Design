#!/usr/bin/env python3
"""Merge data-parallel MT-AIME2024 shards and report anti-forgetting metrics."""

import argparse
import glob
import json
from pathlib import Path


def load_run(name, prefix):
    files = sorted(glob.glob(f"{prefix}_shard*.json"))
    if not files:
        print(f"[WARN] No shard files found for {name}: {prefix}_shard*.json")
        return None

    total_correct = 0
    total_solutions = 0
    total_problems = 0
    language_counts = {}

    for file_name in files:
        with open(file_name, encoding="utf-8") as f:
            shard = json.load(f)
        if shard.get("dataset") != "mt-aime2024":
            raise ValueError(f"Unexpected dataset in {file_name}: {shard.get('dataset')}")

        total_correct += shard["average_at_n"]
        total_solutions += shard["total_solutions"]
        total_problems += shard["num_problems"]
        for language, metrics in shard.get("language_metrics", {}).items():
            aggregate = language_counts.setdefault(
                language, {"num_problems": 0, "correct": 0, "total_solutions": 0}
            )
            aggregate["num_problems"] += metrics["num_problems"]
            aggregate["correct"] += metrics["correct"]
            aggregate["total_solutions"] += metrics["total_solutions"]

    if total_solutions == 0:
        raise ValueError(f"Run {name} contains zero evaluated solutions")

    per_language = {}
    for language, counts in sorted(language_counts.items()):
        counts["average_at_n_pct"] = counts["correct"] / counts["total_solutions"] * 100
        per_language[language] = counts

    english = per_language.get("en", {}).get("average_at_n_pct")
    non_english_scores = [
        metrics["average_at_n_pct"]
        for language, metrics in per_language.items()
        if language != "en"
    ]
    non_english_macro = (
        sum(non_english_scores) / len(non_english_scores) if non_english_scores else None
    )
    language_gap = (
        english - non_english_macro
        if english is not None and non_english_macro is not None
        else None
    )

    return {
        "num_shards": len(files),
        "num_problems": total_problems,
        "correct": total_correct,
        "total_solutions": total_solutions,
        "overall_average_at_n_pct": total_correct / total_solutions * 100,
        "english_average_at_n_pct": english,
        "non_english_macro_average_at_n_pct": non_english_macro,
        "language_gap_pct": language_gap,
        "per_language": per_language,
    }


def add_comparisons(results):
    base = results.get("base")
    l1 = next((metrics for name, metrics in results.items() if name.startswith("L1-step")), None)
    for name, metrics in results.items():
        score = metrics["overall_average_at_n_pct"]
        if base is not None:
            base_score = base["overall_average_at_n_pct"]
            metrics["ood_change_vs_base"] = score - base_score
            metrics["forgetting_vs_base"] = base_score - score
        if l1 is not None:
            metrics["gain_over_l1_step200"] = score - l1["overall_average_at_n_pct"]


def format_optional(value):
    return "n/a" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PREFIX",
        help="Run name and shard prefix, e.g. base=/logs/base",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {}
    for run_spec in args.run:
        if "=" not in run_spec:
            raise ValueError(f"Invalid --run value: {run_spec}; expected NAME=PREFIX")
        name, prefix = run_spec.split("=", 1)
        metrics = load_run(name, prefix)
        if metrics is not None:
            results[name] = metrics

    if not results:
        raise RuntimeError("No completed MT-AIME2024 runs were found")

    add_comparisons(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nMT-AIME2024 Avg@N summary")
    print(
        f"{'Run':<14} {'Overall':>8} {'English':>8} {'Non-En':>8} "
        f"{'Gap':>8} {'vs Base':>8} {'vs L1':>8}"
    )
    for name, metrics in results.items():
        print(
            f"{name:<14} "
            f"{metrics['overall_average_at_n_pct']:>8.2f} "
            f"{format_optional(metrics['english_average_at_n_pct']):>8} "
            f"{format_optional(metrics['non_english_macro_average_at_n_pct']):>8} "
            f"{format_optional(metrics['language_gap_pct']):>8} "
            f"{format_optional(metrics.get('ood_change_vs_base')):>8} "
            f"{format_optional(metrics.get('gain_over_l1_step200')):>8}"
        )
    print(f"\nDetailed summary: {output}")


if __name__ == "__main__":
    main()
