# Training data

`privileged_contexts_l1_l5.jsonl` is derived from the public OPSD pool
[`siyanzhao/Openthoughts_math_30k_opsd`](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd).
It contains 29,434 rows in the exact training order.

Each row contains `sample_id`, `source`, `problem`, `L1_full_solution`,
`L2_strategy`, `L3_framing`, `L4_category`, and `L5_answer_only`. L2--L4 are
the primary Qwen3.5-397B-A17B compiler outputs. L5 preserves the exact training
context, including seven proof-target overrides implemented in
`hint_generation/proof_answer_overrides.py`.

The file removes duplicated metadata from the five source Parquet datasets.
Its checksum is recorded in the root `CHECKSUMS.sha256`.

Public availability does not by itself establish redistribution rights. The
local OPSD snapshot contained no dataset license record, so users should consult
the upstream dataset card and underlying-source terms. Original source text may
contain names, external links or contact information unrelated to the authors
of this release.
