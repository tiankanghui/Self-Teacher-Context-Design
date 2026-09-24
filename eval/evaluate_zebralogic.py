#!/usr/bin/env python3
"""Evaluate full-grid constraint reasoning on ZebraLogic grid mode."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vllm import SamplingParams

from evaluate_math import load_vllm_model
from zebralogic_data import load_zebralogic


def build_prompt(record):
    attributes = record["header"][1:]
    house_keys = [f"House {index}" for index in range(1, len(record["solution_rows"]) + 1)]
    skeleton = {
        "solution": {
            house: {attribute: "<value>" for attribute in attributes}
            for house in house_keys
        }
    }
    return (
        "Solve the following logic-grid puzzle. Determine every attribute for every house.\n\n"
        f"{record['puzzle']}\n\n"
        "Return only one valid JSON object in your final answer, with exactly this structure:\n"
        f"{json.dumps(skeleton, ensure_ascii=False)}\n"
        f"Use exactly these house keys: {json.dumps(house_keys, ensure_ascii=False)}.\n"
        f"Use exactly these attribute keys: {json.dumps(attributes, ensure_ascii=False)}.\n"
        "Copy attribute values exactly as written in the puzzle. Include every house and every cell. "
        "Do not use Markdown fences or add prose outside the JSON object."
    )


def extract_solution(text):
    """Extract the last complete JSON object carrying the official `solution` key."""
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in reversed(starts):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("solution"), dict):
            return value["solution"], "json_solution"
    return None, None


def normalize_cell(value):
    # The official evaluator uses the first item if a model emits a singleton list.
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    return str(value).strip().casefold()


def score_solution(prediction, gold):
    correct_cells = 0
    total_cells = 0
    for house, attributes in gold.items():
        predicted_house = prediction.get(house, {}) if isinstance(prediction, dict) else {}
        if not isinstance(predicted_house, dict):
            predicted_house = {}
        for attribute, gold_value in attributes.items():
            total_cells += 1
            predicted_value = predicted_house.get(attribute)
            correct_cells += int(normalize_cell(predicted_value) == normalize_cell(gold_value))
    return correct_cells, total_cells, correct_cells == total_cells


def percentage(numerator, denominator):
    return numerator / denominator * 100 if denominator else None


def compact_metrics(results):
    buckets = {
        "by_size": defaultdict(lambda: {"puzzles": 0, "solved": 0, "cells": 0, "correct_cells": 0}),
        "by_difficulty": defaultdict(lambda: {"puzzles": 0, "solved": 0, "cells": 0, "correct_cells": 0}),
        "by_scale": defaultdict(lambda: {"puzzles": 0, "solved": 0, "cells": 0, "correct_cells": 0}),
    }
    parsed = 0
    length_cutoffs = 0
    solved = 0
    correct_cells = 0
    cells = 0
    for row in results:
        parsed += int(row["parsed"])
        length_cutoffs += int(row["finish_reason"] == "length")
        solved += int(row["solved"])
        correct_cells += row["correct_cells"]
        cells += row["total_cells"]
        for bucket_name, key in (
            ("by_size", row["size"]),
            ("by_difficulty", row["difficulty"]),
            ("by_scale", row["scale_group"]),
        ):
            stats = buckets[bucket_name][key]
            stats["puzzles"] += 1
            stats["solved"] += int(row["solved"])
            stats["cells"] += row["total_cells"]
            stats["correct_cells"] += row["correct_cells"]

    for bucket in buckets.values():
        for stats in bucket.values():
            stats["puzzle_accuracy_pct"] = percentage(stats["solved"], stats["puzzles"])
            stats["cell_accuracy_pct"] = percentage(stats["correct_cells"], stats["cells"])
    num_puzzles = len(results)
    return {
        "solved": solved,
        "puzzle_accuracy_pct": percentage(solved, num_puzzles),
        "correct_cells": correct_cells,
        "total_cells": cells,
        "cell_accuracy_pct": percentage(correct_cells, cells),
        "parsed": parsed,
        "no_answer": num_puzzles - parsed,
        "no_answer_rate": percentage(num_puzzles - parsed, num_puzzles),
        "length_cutoffs": length_cutoffs,
        "length_cutoff_rate": percentage(length_cutoffs, num_puzzles),
        **{name: dict(values) for name, values in buckets.items()},
    }


def create_lora_request(checkpoint_dir):
    if checkpoint_dir is None:
        return None
    from vllm.lora.request import LoRARequest

    checkpoint = Path(checkpoint_dir)
    if not (checkpoint / "adapter_model.safetensors").exists() and not (
        checkpoint / "adapter_model.bin"
    ).exists():
        raise FileNotFoundError(f"No LoRA adapter weights found in {checkpoint}")
    return LoRARequest("checkpoint_lora", 1, str(checkpoint))


def evaluate(args):
    if args.val_n != 1:
        raise ValueError("ZebraLogic uses one full-grid generation per puzzle; set --val_n 1")
    records = load_zebralogic(
        args.data_dir, check_expected_size=args.num_samples is None
    )
    if args.num_samples is not None:
        records = records[: min(args.num_samples, len(records))]
    full_num_records = len(records)

    if args.num_shards > 1:
        shard_size = (full_num_records + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, full_num_records)
        records = records[start:end]
        print(
            f"Shard {args.shard_id}/{args.num_shards}: global puzzles "
            f"{start}-{end - 1} ({len(records)} puzzles)"
        )
    if not records:
        raise ValueError("Selected ZebraLogic shard is empty")

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )
    lora_request = create_lora_request(args.checkpoint_dir)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(record)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for record in records
    ]
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
    )
    generate_kwargs = {"use_tqdm": True}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request
    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)
    if len(outputs) != len(records):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(records)} prompts")

    results = []
    for record, output in zip(records, outputs):
        if len(output.outputs) != 1:
            raise RuntimeError("Expected exactly one ZebraLogic generation per prompt")
        candidate = output.outputs[0]
        prediction, extraction_method = extract_solution(candidate.text)
        correct_cells, total_cells, solved = score_solution(
            prediction or {}, record["solution_table"]
        )
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            finish_reason = str(finish_reason)
        results.append(
            {
                "problem_id": record["problem_id"],
                "source_row": record["source_row"],
                "size": record["size"],
                "difficulty": record["difficulty"],
                "scale_group": record["scale_group"],
                "parsed": prediction is not None,
                "extraction_method": extraction_method,
                "correct_cells": correct_cells,
                "total_cells": total_cells,
                "cell_accuracy_pct": percentage(correct_cells, total_cells),
                "solved": solved,
                "finish_reason": finish_reason,
                "predicted_solution": prediction,
                "ground_truth": record["solution_table"],
                "full_generation": candidate.text,
            }
        )

    metrics = compact_metrics(results)
    summary = {
        "base_model": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset": "zebralogic-grid",
        "split": "test",
        "protocol": "zero-shot-full-grid-json",
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "val_n": 1,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "num_problems": len(results),
        **metrics,
        "results": results,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nZebraLogic grid-mode shard results")
    print(f"  Puzzles: {len(results)}")
    print(f"  Puzzle accuracy: {metrics['puzzle_accuracy_pct']:.2f}%")
    print(f"  Cell accuracy: {metrics['cell_accuracy_pct']:.2f}%")
    print(f"  No-answer rate: {metrics['no_answer_rate']:.2f}%")
    print(f"  Length-cutoff rate: {metrics['length_cutoff_rate']:.2f}%")
    print(f"  Output: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--val_n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
