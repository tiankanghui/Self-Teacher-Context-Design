# Context construction

The paper uses a staged compiler: L4 (category) is generated first, L3
(framing) conditions on L4, and L2 (strategy) conditions on both L4 and L3.

| File | Purpose |
|---|---|
| `dim1_abstraction_level_v2.py` | Exact staged prompts and resumable compiler |
| `llm_client.py` | OpenAI-compatible endpoint client |
| `audit_dim1_output.py` | Row alignment and output-format validation |
| `build_l5_answer_only.py` | L5 answer-only construction |
| `proof_answer_overrides.py` | Seven proof-target overrides |
| `merge_hints.py` | Compatibility conversion to the original Parquet layout |

Run `scripts/generate_contexts.sh` from the repository root. L1 comes directly
from the reference solution; L5 is constructed without an LLM call.
