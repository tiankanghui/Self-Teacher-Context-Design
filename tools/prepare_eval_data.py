#!/usr/bin/env python3
"""Install and verify the seven public benchmarks used in the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "aime24": "aime2024/train.parquet",
    "aime25": "aime2025/train.parquet",
    "hmmt25": "hmmt2025/train.parquet",
    "mt_aime": "mt-aime2024/mt-aime2024.parquet",
    "gpqa": "gpqa_diamond/gpqa_diamond.parquet",
    "autologi": "autologi/AutoLogi_en.jsonl",
    "zebralogic": "ZebraLogicBench/grid_mode/test-00000-of-00001.parquet",
}
ID_SOURCES = {
    "aime24": "HuggingFaceH4/aime_2024",
    "aime25": "yentinglin/aime_2025",
    "hmmt25": "MathArena/hmmt_feb_2025",
}


def fingerprint(path: Path) -> tuple[int, str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    else:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return len(rows), digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/eval")
    parser.add_argument("--download-id", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    for name in FILES:
        parser.add_argument(f"--{name.replace('_', '-')}-file", dest=name, type=Path)
    args = parser.parse_args()

    expected = json.loads((ROOT / "manifests/evaluation_data.json").read_text())
    if args.verify_only and (
        args.download_id or any(getattr(args, name) for name in FILES)
    ):
        parser.error("--verify-only cannot be combined with downloads or input files")

    installed = 0
    for name, relative in FILES.items():
        target = args.output_dir / relative
        source = getattr(args, name)
        download = args.download_id and name in ID_SOURCES
        if source and download:
            option = name.replace("_", "-")
            parser.error(f"Choose either --download-id or --{option}-file")
        if source or download:
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"{target} exists; pass --overwrite")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source:
                if not source.is_file():
                    raise FileNotFoundError(source)
                shutil.copyfile(source, target)
            else:
                from datasets import load_dataset

                load_dataset(ID_SOURCES[name], split="train").to_parquet(str(target))
        if not target.exists():
            if args.require_all:
                raise FileNotFoundError(f"Missing benchmark: {name} ({relative})")
            print(f"Not installed: {name} ({relative})")
            continue
        count, digest = fingerprint(target)
        reference = expected[name]
        if count != reference["rows"]:
            raise ValueError(f"{name}: expected {reference['rows']} rows, got {count}")
        matches = digest == reference["content_sha256"]
        print(f"{name}: {count} rows; paper snapshot match={matches}")
        if args.strict and not matches:
            raise ValueError(f"{name}: content differs from the paper snapshot")
        installed += 1
    if installed == 0:
        raise ValueError("No benchmark files found or installed")


if __name__ == "__main__":
    main()
