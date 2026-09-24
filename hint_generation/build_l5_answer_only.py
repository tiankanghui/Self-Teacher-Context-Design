"""Build the L5 answer-only OPSD training parquet.

L5 is an answer anchor, not another point on the L1--L4 abstraction axis.  The
student still sees only the problem; the teacher's ``solution`` field contains
only the original final answer.  All other columns and the original row order
are preserved.

Seven proof problems in the released training parquet have an empty ``Answer``
field.  For those rows we insert only the proposition being proved--never the
reference derivation.  The overrides are keyed by a SHA-256 hash of the exact
problem text so that a reordered dataset cannot silently attach an answer to
the wrong problem.
"""

import argparse
import glob
import hashlib
import os

import pandas as pd

from proof_answer_overrides import PROOF_ANSWER_OVERRIDES


def problem_hash(problem):
    return hashlib.sha256(str(problem).encode("utf-8")).hexdigest()


def clean_answer(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser(description="Build answer-only L5 training data")
    parser.add_argument(
        "--parquet",
        required=True,
        help="Glob for the original training parquet shards",
    )
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--answer-column", default="Answer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(args.parquet))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched: {args.parquet}")
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}; pass --overwrite to replace it"
        )

    frames = [pd.read_parquet(path) for path in parquet_files]
    frame = pd.concat(frames, ignore_index=True)
    required = {"problem", "solution", args.answer_column}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    original_problems = frame["problem"].copy()
    answers = []
    override_count = 0
    unresolved = []
    used_override_hashes = set()

    for index, row in frame.iterrows():
        answer = clean_answer(row[args.answer_column])
        if not answer:
            digest = problem_hash(row["problem"])
            answer = PROOF_ANSWER_OVERRIDES.get(digest, "")
            if not answer:
                unresolved.append((index, digest, str(row["problem"])[:160]))
                continue
            override_count += 1
            used_override_hashes.add(digest)
        answers.append(answer)

    if unresolved:
        preview = "\n".join(str(item) for item in unresolved[:20])
        raise ValueError(f"Unresolved empty answers ({len(unresolved)}):\n{preview}")
    if used_override_hashes != set(PROOF_ANSWER_OVERRIDES):
        unused = sorted(set(PROOF_ANSWER_OVERRIDES) - used_override_hashes)
        raise ValueError(f"Expected proof-answer overrides were not used: {unused}")
    if len(answers) != len(frame):
        raise RuntimeError(f"Answer count mismatch: {len(answers)} != {len(frame)}")

    # The trainer reads only problem/solution.  Preserve every other column for
    # exact compatibility with the existing L2--L4 datasets.
    frame["solution"] = answers
    if frame["solution"].isna().any() or frame["solution"].str.strip().eq("").any():
        raise RuntimeError("L5 output contains an empty solution")
    if not frame["problem"].equals(original_problems):
        raise RuntimeError("Problem order changed while building L5")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    frame.to_parquet(args.output, index=False)

    # Reload the materialized file instead of trusting only the in-memory frame.
    check = pd.read_parquet(args.output)
    if len(check) != len(frame):
        raise RuntimeError(f"Reloaded row count mismatch: {len(check)} != {len(frame)}")
    if list(check.columns) != list(frame.columns):
        raise RuntimeError("Reloaded L5 columns differ from the source columns")
    if not check["problem"].equals(original_problems.reset_index(drop=True)):
        raise RuntimeError("Reloaded L5 problem order differs from the source")
    if not check["solution"].equals(pd.Series(answers, name="solution")):
        raise RuntimeError("Reloaded L5 answers differ from the constructed answers")

    lengths = check["solution"].str.len()
    print(f"Loaded {len(frame)} rows from {len(parquet_files)} source shard(s)")
    print(f"Answer-column rows: {len(frame) - override_count}")
    print(f"Proof-target overrides: {override_count}")
    print(
        "Answer characters: "
        f"min={int(lengths.min())} median={float(lengths.median()):.1f} "
        f"mean={float(lengths.mean()):.1f} max={int(lengths.max())}"
    )
    print(f"Saved verified L5 answer-only parquet: {args.output}")


if __name__ == "__main__":
    main()
