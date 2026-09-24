#!/usr/bin/env python3
"""Generate and programmatically verify open-ended AutoLogi solutions."""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from vllm import SamplingParams

from evaluate_math import load_vllm_model


VERIFY_FUNCTION_CODE = """
def verify_function(inputs, inputs_check, constraint_list):
    if not inputs_check(inputs):
        return False
    for constraint in constraint_list:
        if not constraint(inputs):
            return False
    return True
"""


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"idx", "prompt", "code", "group_id"}
            missing = required - set(record)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            code = record["code"]
            for code_key in ("Inputs_Check_code", "Constraint_List_code"):
                if code_key not in code:
                    raise ValueError(f"{path}:{line_number} missing code.{code_key}")
            records.append(record)
    return records


def load_autologi(eval_data_path, split):
    data_dir = Path(eval_data_path) / "autologi"
    split_files = {
        "en": data_dir / "AutoLogi_en.jsonl",
        "cn": data_dir / "AutoLogi_cn.jsonl",
    }
    selected_splits = ["en", "cn"] if split == "both" else [split]
    records = []
    for selected_split in selected_splits:
        path = split_files[selected_split]
        if not path.is_file():
            raise FileNotFoundError(f"AutoLogi file not found: {path}")
        split_records = load_jsonl(path)
        for record in split_records:
            record["source_split"] = selected_split
            record["problem_id"] = f"{selected_split}:{record['idx']}"
        records.extend(split_records)
        print(f"Loaded AutoLogi-{selected_split.upper()}: {len(split_records)} examples from {path}")
    return records


def extract_code(text):
    pattern = re.compile(r"```(?:\s*[\w]+)?\s*\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    return blocks[-1] if blocks else None


def extract_last_balanced(text, left, right):
    stack = []
    start = None
    candidates = []
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == left:
            if not stack:
                start = index
            stack.append(char)
        elif char == right and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates[-1] if candidates else None


def parse_model_answer(text):
    candidate = extract_code(text)
    if candidate is None:
        last_dict = extract_last_balanced(text, "{", "}")
        last_list = extract_last_balanced(text, "[", "]")
        if last_dict is not None:
            candidate = last_dict
        elif last_list is not None:
            candidate = last_list
        else:
            return None, "extraction_failed"

    candidate = re.sub(r"//.*?\n", "", candidate).strip()
    variants = [candidate]
    if "=" in candidate:
        variants.append(candidate.split("=", 1)[1].strip())
    for variant in variants:
        normalized = re.sub(r"\btrue\b", "True", variant, flags=re.IGNORECASE)
        normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.IGNORECASE)
        try:
            return ast.literal_eval(normalized), None
        except (SyntaxError, ValueError, TypeError):
            try:
                return json.loads(variant), None
            except (json.JSONDecodeError, TypeError):
                pass
    return None, "extraction_failed"


def count_constraints(constraint_code):
    try:
        tree = ast.parse(constraint_code)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "constraint_list" for target in targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return len(node.value.elts)
    except SyntaxError:
        pass
    return None


def run_verifier(params, code, timeout_seconds):
    program = "\n".join(
        [
            code["Inputs_Check_code"],
            code["Constraint_List_code"],
            VERIFY_FUNCTION_CODE,
            f"result = verify_function({params!r}, inputs_check, constraint_list)",
            "print(result)",
        ]
    )
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="autologi_verify_") as temp_dir:
            program_path = Path(temp_dir) / "verify.py"
            program_path.write_text(program, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", str(program_path)],
                cwd=temp_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, f"verifier_launch_error: {exc}"

    if completed.returncode != 0:
        error = completed.stderr.strip().replace("\n", " ")
        return False, f"verifier_error: {error[:1000]}"
    if completed.stdout.strip() == "True":
        return True, None
    return False, "constraint_failed"


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
    records = load_autologi(args.eval_data_path, args.autologi_split)
    if args.num_samples is not None:
        records = records[: min(args.num_samples, len(records))]

    if args.num_shards > 1:
        shard_size = (len(records) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(records))
        records = records[start:end]
        print(
            f"Shard {args.shard_id}/{args.num_shards}: global examples "
            f"{start}-{end - 1} ({len(records)} examples)"
        )
    if not records:
        raise ValueError("Selected AutoLogi shard is empty")

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
            [{"role": "user", "content": record["prompt"]}],
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
    pass_at_n = 0
    extraction_failures = 0
    verifier_errors = 0
    timeouts = 0
    by_constraint_count = defaultdict(lambda: {"num_problems": 0, "correct": 0, "total": 0})

    for record, output in zip(records, outputs):
        generations = []
        correct_for_problem = 0
        constraint_count = count_constraints(record["code"]["Constraint_List_code"])
        for candidate in output.outputs:
            params, parse_error = parse_model_answer(candidate.text)
            if parse_error is not None:
                correct = False
                error = parse_error
                extraction_failures += 1
            else:
                correct, error = run_verifier(params, record["code"], args.verifier_timeout)
                if error == "timeout":
                    timeouts += 1
                elif error is not None and error.startswith("verifier_"):
                    verifier_errors += 1
            correct_for_problem += int(correct)
            generations.append(
                {
                    "full_generation": candidate.text,
                    "parsed_answer": repr(params) if parse_error is None else None,
                    "correct": correct,
                    "error": error,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                }
            )

        total_correct += correct_for_problem
        has_correct = correct_for_problem > 0
        pass_at_n += int(has_correct)
        bucket = by_constraint_count[str(constraint_count)]
        bucket["num_problems"] += 1
        bucket["correct"] += correct_for_problem
        bucket["total"] += args.val_n
        results.append(
            {
                "problem_id": record["problem_id"],
                "idx": record["idx"],
                "group_id": record["group_id"],
                "source_split": record["source_split"],
                "constraint_count": constraint_count,
                "prompt": record["prompt"],
                "val_n": args.val_n,
                "num_correct": correct_for_problem,
                "pass_at_n": has_correct,
                "generations": generations,
            }
        )

    num_problems = len(results)
    total_solutions = num_problems * args.val_n
    average_at_n_pct = total_correct / total_solutions * 100
    pass_at_n_pct = pass_at_n / num_problems * 100
    for bucket in by_constraint_count.values():
        bucket["accuracy_pct"] = bucket["correct"] / bucket["total"] * 100

    summary = {
        "base_model": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset": "autologi",
        "autologi_split": args.autologi_split,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "val_n": args.val_n,
        "num_problems": num_problems,
        "total_solutions": total_solutions,
        "average_at_n": total_correct,
        "average_at_n_pct": average_at_n_pct,
        "pass_at_n": pass_at_n,
        "pass_at_n_pct": pass_at_n_pct,
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failures / total_solutions * 100,
        "verifier_errors": verifier_errors,
        "verifier_error_rate": verifier_errors / total_solutions * 100,
        "verifier_timeouts": timeouts,
        "verifier_timeout_rate": timeouts / total_solutions * 100,
        "by_constraint_count": dict(sorted(by_constraint_count.items(), key=lambda item: int(item[0]) if item[0] != "None" else -1)),
        "results": results,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nAutoLogi results")
    print(f"  Split: {args.autologi_split}")
    print(f"  Problems: {num_problems}")
    print(f"  Accuracy/Avg@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"  Pass@{args.val_n}: {pass_at_n_pct:.2f}%")
    print(f"  Extraction failure rate: {summary['extraction_failure_rate']:.2f}%")
    print(f"  Verifier error rate: {summary['verifier_error_rate']:.2f}%")
    print(f"  Output: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--eval_data_path", required=True)
    parser.add_argument("--autologi_split", choices=["en", "cn", "both"], default="en")
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
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--verifier_timeout", type=int, default=20)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
