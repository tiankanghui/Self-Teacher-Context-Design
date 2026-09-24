#!/usr/bin/env python3
"""Evaluate fixed-choice GPQA-Diamond with repeated sampled accuracy."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from vllm import SamplingParams

from evaluate_math import load_vllm_model


FINAL_ANSWER_INSTRUCTION = """

Please reason step by step. On the final line, write exactly "Answer: X", where X is one of A, B, C, or D.
"""


def normalize_answer(answer):
    normalized = str(answer).strip().upper()
    match = re.fullmatch(r"(?:ANSWER\s*:\s*)?\(?([ABCD])\)?[\s\.]?", normalized)
    if not match:
        raise ValueError(f"Invalid GPQA answer label: {answer!r}")
    return match.group(1)


def extract_answer(text):
    patterns = [
        r"(?i)(?:final\s+answer|answer)\s*(?:is|:|：)\s*\(?([ABCD])\)?",
        r"\\boxed\s*\{\s*([ABCD])\s*\}",
    ]
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), match.group(1).upper()))
    return max(matches)[1] if matches else None


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


def load_gpqa(eval_data_path):
    data_file = Path(eval_data_path) / "gpqa_diamond" / "gpqa_diamond.parquet"
    if not data_file.is_file():
        raise FileNotFoundError(f"GPQA-Diamond parquet not found: {data_file}")
    dataset = load_dataset(
        "parquet", data_files={"test": str(data_file)}, split="test"
    )
    required = {"question", "answer"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"GPQA-Diamond is missing columns: {sorted(missing)}")

    records = []
    for index, example in enumerate(dataset):
        question = str(example["question"]).strip()
        if not question:
            raise ValueError(f"Empty GPQA question at row {index}")
        records.append(
            {
                "problem_id": index,
                "question": question,
                "answer": normalize_answer(example["answer"]),
            }
        )
    if len(records) != 198:
        raise ValueError(f"Expected 198 GPQA-Diamond rows, found {len(records)}")
    print(f"Loaded GPQA-Diamond: {len(records)} questions from {data_file}")
    print(f"Answer distribution: {dict(sorted(Counter(r['answer'] for r in records).items()))}")
    return records


def evaluate(args):
    records = load_gpqa(args.eval_data_path)
    if args.num_samples is not None:
        records = records[: min(args.num_samples, len(records))]

    if args.num_shards > 1:
        shard_size = (len(records) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(records))
        records = records[start:end]
        print(
            f"Shard {args.shard_id}/{args.num_shards}: global questions "
            f"{start}-{end - 1} ({len(records)} questions)"
        )
    if not records:
        raise ValueError("Selected GPQA shard is empty")

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
        distributed_executor_backend=args.distributed_executor_backend,
    )
    lora_request = create_lora_request(args.checkpoint_dir)
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": record["question"] + FINAL_ANSWER_INSTRUCTION,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for record in records
    ]
    sampling_params = SamplingParams(
        n=args.val_n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
    )
    if lora_request is None:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    else:
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_request, use_tqdm=True
        )

    results = []
    total_correct = 0
    extraction_failures = 0
    pass_at_n = 0
    majority_vote_correct = 0
    for record, output in zip(records, outputs):
        generations = []
        predictions = []
        num_correct = 0
        for candidate in output.outputs:
            prediction = extract_answer(candidate.text)
            correct = prediction == record["answer"]
            if prediction is None:
                extraction_failures += 1
            else:
                predictions.append(prediction)
            num_correct += int(correct)
            generations.append(
                {
                    "predicted_answer": prediction,
                    "correct": correct,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "full_generation": candidate.text,
                }
            )

        majority_prediction = Counter(predictions).most_common(1)[0][0] if predictions else None
        majority_correct = majority_prediction == record["answer"]
        total_correct += num_correct
        pass_at_n += int(num_correct > 0)
        majority_vote_correct += int(majority_correct)
        results.append(
            {
                "problem_id": record["problem_id"],
                "question": record["question"],
                "ground_truth": record["answer"],
                "val_n": args.val_n,
                "num_correct": num_correct,
                "pass_at_n": num_correct > 0,
                "majority_prediction": majority_prediction,
                "majority_vote_correct": majority_correct,
                "generations": generations,
            }
        )

    num_problems = len(results)
    total_solutions = num_problems * args.val_n
    summary = {
        "base_model": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset": "gpqa-diamond",
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "val_n": args.val_n,
        "num_problems": num_problems,
        "total_solutions": total_solutions,
        "average_at_n": total_correct,
        "average_at_n_pct": total_correct / total_solutions * 100,
        "pass_at_n": pass_at_n,
        "pass_at_n_pct": pass_at_n / num_problems * 100,
        "majority_vote_at_n": majority_vote_correct,
        "majority_vote_at_n_pct": majority_vote_correct / num_problems * 100,
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failures / total_solutions * 100,
        "results": results,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nGPQA-Diamond results")
    print(f"  Problems: {num_problems}")
    print(f"  Avg@{args.val_n}: {summary['average_at_n_pct']:.2f}%")
    print(f"  Pass@{args.val_n}: {summary['pass_at_n_pct']:.2f}%")
    print(f"  Majority@{args.val_n}: {summary['majority_vote_at_n_pct']:.2f}%")
    print(f"  Extraction failure rate: {summary['extraction_failure_rate']:.2f}%")
    print(f"  Output: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--eval_data_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--val_n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--distributed_executor_backend",
        choices=["auto", "uni", "mp", "ray"],
        default="auto",
    )
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
