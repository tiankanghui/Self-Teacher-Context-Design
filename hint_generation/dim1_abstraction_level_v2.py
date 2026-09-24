"""Compile the paper's L2--L4 semantic contexts.

The staged procedure generates L4 (category), then L3 (framing, conditioned on
L4), then L2 (strategy, conditioned on L4 and L3). The source problem and
reference solution are supplied at each stage. The same prompts are used for
the primary compiler and the smaller-compiler replication.

Prompts give soft typical-length guidance and prohibit the final answer;
realized compliance is measured separately by the semantic audit. The pipeline
retries exceptions, preserves input order, validates references before API
calls, and checks input/compiler metadata before resuming. L1 and L5 are
prepared separately.

From the repository root, run: bash scripts/generate_contexts.sh
"""

import json
import argparse
import hashlib
import sys
import os
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from statistics import mean, median, stdev
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from llm_client import LLMClient


# ═════════════════════════════════════════════════════════════════════════════
#  L4 — Problem Category
#  v3.1: soft length guidance; hardened "via X" ban preserved from v3
# ═════════════════════════════════════════════════════════════════════════════
PROMPT_L4 = """You are generating a "Problem Category" (Level 4) hint for a math problem.

## Strict definition
A one-sentence classification stating:
  1. Which mathematical **domain** the problem belongs to (e.g., analytic geometry, number theory, combinatorics, elementary algebra).
  2. Which **core concepts / objects** are involved (e.g., line equations, prime factorization, recurrence relations).

## HARD constraints — the hint MUST NOT contain
- Any verb suggesting a solution action
  (e.g., "solve", "compute", "prove", "derive", "convert", "use", "apply",
   "perform", "calculate", "count", "combine", "find", "determine")
- Any method / technique / theorem name
- Any thinking pattern or cognitive move
- Any concrete numerical value or the final answer
- Any prepositional phrase that introduces a method or approach:
    forbidden prepositions when followed by a technique-name:
      "via", "using", "through", "by", "with", "involving", "based on",
      "relying on", "under", "employing", "leveraging"
  Example of FORBIDDEN wording:
      "…problem X via inclusion-exclusion."
      "…problem X using Vieta's formulas."

## Format enforcement (mandatory)
Follow the exact pattern:
    "<Domain>: <noun phrase describing the object / concept type>."
where <noun phrase> describes WHAT is being asked, NOT how to solve it.

## Length guidance
A well-formed L4 is typically 10-30 tokens — just long enough to name the
domain and characterize the object. Adapt naturally to the problem's
information density: a bounded-integer counting problem may need only
12 tokens; a multi-object geometric configuration may need 35.

DO NOT pad L4 to any target length by:
  - Adding solution approaches or method hints
  - Adding cognitive strategies
  - Restating the problem's numerical constraints in prose

If you cannot classify the problem in fewer than 40 tokens without hinting
at a method, prefer to STAY SHORT — brevity is a feature of L4, not a flaw.

## Boundary tests (self-check before output)

### Test A — Verb test
Does your sentence contain ANY of these action verbs (as -ing, past, or infinitive)?
    solve, compute, prove, derive, convert, use, apply, perform, calculate,
    count, combine, subtract, find, determine, transform, optimize
- If YES → your sentence describes an action, not a classification. REWRITE.

### Test B — "via X" removal test
Look for any clause introduced by:
    "via / using / through / by / with / involving / based on / employing"
that is followed by a technique/theorem/algorithm name.
- If such a clause exists → DELETE that clause.
  * If the remaining sentence still stands as a complete classification
    → GOOD. Output the cleaned version.
  * If not → the deleted clause was carrying the classification (you were
    classifying by *method*, not by *object*). REWRITE from scratch.

### Test C — Method-name test
Does your sentence contain any specific technique/theorem/algorithm name?
- If YES → you have leaked L2 into L4. REWRITE.

## Examples

Problem: "Find the equation of the line through A(3,-2) with direction (-5,3)."
✓ GOOD L4: "Analytic geometry: line equation from a point and a direction vector."
✗ BAD L4 (action verb "deriving"):
   "Analytic geometry: deriving a line equation from a point and a direction vector."

Problem: "Given real a,b,c and λ>0 such that x^3 + ax^2 + bx + c has three real roots
with x2-x1=λ and x3 > (x1+x2)/2, find max of (2a^3+27c-9ab)/λ^3."
✓ GOOD L4: "Polynomial algebra: extremal value of a coefficient expression under constraints on the roots of a cubic."
✗ BAD L4 (method names via "via X"):
   "Polynomial algebra: optimization of a coefficient expression via root transformation and Vieta's formulas."

Problem: "How many (x,y,z) with 1≤x,y,z≤6 have product divisible by 10?"
✓ GOOD L4: "Combinatorics: counting integer tuples whose product satisfies a divisibility constraint."
✗ BAD L4 (method name via "via inclusion-exclusion"):
   "Combinatorics: counting tuples with product divisibility via inclusion-exclusion."

## Task

Problem:
{problem}

Reference solution (for your understanding only):
{solution}

Output ONLY the L4 hint text, no explanation, no quotes, no markdown fences."""


# ═════════════════════════════════════════════════════════════════════════════
#  L3 — Meta-Strategy (sees L4)
#  v3.1: soft length guidance; hardened cognitive-vs-operation distinction preserved
# ═════════════════════════════════════════════════════════════════════════════
PROMPT_L3 = """You are generating a "Meta-Strategy" (Level 3) hint for a math problem.

## Context
The L4 (Problem Category) for this problem has already been generated:
    "{L4}"

You must now generate the L3 (Meta-Strategy), which sits one level MORE SPECIFIC than L4
but STRICTLY MORE ABSTRACT than any named method (which would be L2).

## Strict definition
A meta-strategy describes a **cognitive framing shift** — a single mental move
that reframes the problem, WITHOUT specifying any concrete method or step sequence.

## Cognitive-strategy vs Operation-sequence distinction  [CRITICAL]

L3 must be a COGNITIVE STRATEGY — a single framing shift.
L3 must NOT be an OPERATION SEQUENCE — a step-by-step recipe.

### ✓ Cognitive strategies (VALID L3)
- "shift perspective to the complement"
- "unify heterogeneous quantities into a common representation"
- "exchange the roles of input and output"
- "seek an invariant under transformation"
- "translate constraints from one representation to another"
- "reduce to a simpler / already-solved form"
- "consider extreme or boundary cases first"
- "decouple the variables so that they can be treated independently"

### ✗ Operation sequences (INVALID — this is L2's recipe in disguise)
- "count the complement, then subtract from the total" (this is IEX's recipe)
- "align their exponents so the bases can be directly compared" (this is power-of-a-power's recipe)
- "identify missing prime factors, then combine these exclusion cases" (again IEX)

## HARD constraints — the hint MUST NOT contain
- Any **specific named method / theorem / algorithm**
  (no "similar triangles", no "induction", no "factoring", no "Vieta",
   no "inclusion-exclusion", no "Tschirnhaus", no "algorithm", no "algebraic sum")
- Any **sequence of concrete verbs** (see Verb-count test below)
- Any concrete numerical value
- Any final answer

## Length guidance
A well-formed L3 is typically 20-60 tokens — long enough to state the
cognitive shift and briefly indicate why this framing applies, but short
enough to remain a single framing statement rather than a plan.

Some problems have very crisp cognitive shifts (e.g., "reframe as its
complement") — those L3 hints can naturally be 10-15 tokens. Others
require more context to make the framing meaningful — those can extend
to 70+ tokens.

Adapt to the natural information density of THIS problem's cognitive shift.
DO NOT pad L3 by:
  - Adding operational steps (which would drift to L2)
  - Naming specific methods (which would drift to L2)
  - Restating the problem

## Boundary tests (self-check before output)

### Test A — Transferability
Could this same sentence be applied UNCHANGED to a different problem that
also falls under L4 = "{L4}" but requires a different specific method?
- If YES → valid L3.
- If NO (prescribes a specific technique) → drifted to L2. REWRITE.

### Test B — Verb-count test
Count the concrete action verbs in your sentence
(e.g. count, subtract, add, align, combine, rewrite, factor,
 substitute, expand, identify, list, compute, apply, transform).
- If verb count ≥ 2 → you are describing an operation SEQUENCE, not a
  single cognitive shift. REWRITE as a single-clause framing statement.
- If verb count is 0 or 1 (typically a framing verb like "shift", "unify",
  "reduce", "translate", "seek", "decouple", "reframe", "reverse",
  "exchange") → valid L3.

### Test C — "then / next / after" ban
Your sentence MUST NOT contain the words "then", "next", "after",
"subsequently", "and then". These signal a sequence of actions.

## Examples

Problem: "Sum 555_6 + 55_6 + 5_6 in base 6."
L4: "Base-b arithmetic: addition within a positional numeral system with non-decimal base."
✓ GOOD L3: "Decompose the problem into local operations at each position, with information flowing between adjacent positions."
✗ BAD L3 (names algorithm): "Execute the standard arithmetic algorithm adapted for a non-decimal base."
✗ BAD L3 (operation sequence): "Add the digits column by column, then propagate the carry to the next column."

Problem: "Count (x,y,z) with 1≤x,y,z≤6, product divisible by 10."
L4: "Combinatorics: counting integer tuples whose product satisfies a divisibility constraint."
✓ GOOD L3: "Shift the counting problem to its complement space, where the divisibility constraint becomes easier to characterize."
✗ BAD L3 (operation sequence — IEX recipe): "Count the complement by identifying missing prime factors, then combine these exclusion cases to subtract from the total."

Problem: "Compare p=2^3009, q=3^2006, r=5^1003."
L4: "Elementary algebra: ordering of exponential terms with distinct bases and exponents."
✓ GOOD L3: "Reframe the terms into a common representational form so that comparison depends on a single varying attribute."
✗ BAD L3 (operation sequence): "Align their exponents by rewriting each term, then compare the resulting bases."

Problem: "Find max of (2a^3+27c-9ab)/λ^3 given cubic root constraints."
L4: "Polynomial algebra: extremal value of a coefficient expression under root constraints."
✓ GOOD L3: "Reduce the coefficient expression to a simpler form by reframing it in terms of the roots themselves."
✗ BAD L3 (two-verb sequence): "Transform the polynomial to simplify its structure, then express the target quantity in terms of the transformed roots."

## Task

Problem:
{problem}

Reference solution (for your understanding only):
{solution}

L4 (for context, one step above L3):
{L4}

Output ONLY the L3 hint text, no explanation, no quotes, no markdown fences."""


# ═════════════════════════════════════════════════════════════════════════════
#  L2 — Strategy Path (sees L4 + L3)
#  v3.1: soft length guidance; boundary tests preserved from v3
# ═════════════════════════════════════════════════════════════════════════════
PROMPT_L2 = """You are generating a "Strategy Path" (Level 2) hint for a math problem.

## Context
The L4 and L3 for this problem have already been generated:
- L4 (Category):       "{L4}"
- L3 (Meta-Strategy):  "{L3}"

You must now generate the L2 (Strategy Path), which sits one level MORE SPECIFIC than L3
but STRICTLY MORE ABSTRACT than the full solution (L1).

## Strict definition
An L2 hint **names the specific method, theorem, or tool** to be used — and stops there.

## What L2 IS
- Names a concrete technique, e.g.:
  * "similar triangles"
  * "the parametric form of a line"
  * "induction on n"
  * "factoring by grouping"
  * "the law of cosines"
  * "columnar addition with base-b carry"
  * "the Principle of Inclusion-Exclusion"
  * "Vieta's formulas"
- MAY add one short clause explaining WHY this method applies
  (at the conceptual level, not procedural).

## HARD constraints — the hint MUST NOT contain
- Any procedural language: "first X, then Y", "starting from...", "step by step",
  numbered steps, or any words like "then", "next", "after that", "finally",
  "write down", "carry over".
  (Exception: "then" is OK when it merely connects two named methods,
  as in "apply Method A, then Vieta's formulas" — but NEVER when it
  connects two procedural steps.)
- Any concrete numerical value from the problem
- Any symbolic substitution (no "let x = ...", no "set t = ...")
- Any intermediate expression or the final answer

## Length guidance
A well-formed L2 is typically 30-90 tokens — enough to name the method(s)
and provide a brief conceptual justification of why they apply.

Adapt to the problem's method complexity:
  - Single-method problems (e.g., "use the law of cosines") may need only
    25-40 tokens.
  - Multi-method problems (e.g., "apply Tschirnhaus + Vieta") may need
    60-100 tokens to describe how the methods connect.
  - Do NOT pad L2 with procedural details to reach a target length.
  - Do NOT compress L2 by dropping the conceptual WHY, if the WHY genuinely
    helps clarify the method choice.

## Boundary tests (self-check before output)

### Test A — Sequence-word ban
Your sentence MUST NOT contain: "then", "next", "after", "first", "finally",
or numbered step markers — EXCEPT when they connect named methods (not steps).
- If any of these appear as procedural connectors → drifted to L1. REWRITE.

### Test B — Named-method requirement
Your sentence MUST name at least one specific technique that a student could
look up (a theorem name, a method name, an algorithm name).
- If NO named method is present → still at L3. REWRITE to explicitly name the method.

## Examples

Problem: "Sum 555_6 + 55_6 + 5_6 in base 6."
L3: "Decompose the problem into local operations at each position, with information flowing between adjacent positions."
✓ GOOD L2: "Apply the standard columnar addition algorithm, using base 6 for both digit-sum reduction and carry propagation between adjacent positions."
✗ BAD L2 (procedural drift to L1):
   "Perform column-wise addition from right to left, converting each column sum into base 6 to determine the digit to write and the carry value for the next position."

Problem: "Count (x,y,z) with product divisible by 10."
L3: "Shift the counting problem to its complement space."
✓ GOOD L2: "Apply the Principle of Inclusion-Exclusion to the union of sets defined by the absence of specific prime factors required for divisibility."

Problem: "Find max of (2a^3+27c-9ab)/λ^3 given cubic root constraints."
L3: "Reduce the coefficient expression to a simpler form by reframing it in terms of the roots themselves."
✓ GOOD L2: "Apply a Tschirnhaus transformation to depress the cubic, then use Vieta's formulas to rewrite the target expression in terms of the transformed roots for single-variable optimization."
   (Note: 'then' here connects two named methods, not procedural steps — acceptable.)

Problem: "Find equation of line through A(3,-2) with direction (-5,3)."
L3: "Translate the geometric constraint into an algebraic relation among coordinates."
✓ GOOD L2: "Use the point-direction (two-point ratio) form of a line, where the displacement vector from the given point must be parallel to the direction vector."

## Task

Problem:
{problem}

Reference solution (for your understanding only; do NOT reproduce its concrete steps or values):
{solution}

L4 (for context):
{L4}

L3 (for context, one step above L2):
{L3}

Output ONLY the L2 hint text, no explanation, no quotes, no markdown fences."""


# ═════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═════════════════════════════════════════════════════════════════════════════
def count_lines(path):
    """Count lines in a file, returns 0 if file doesn't exist."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                raise ValueError(f"Row {index}: blank JSONL line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Row {index}: expected a JSON object")
            data.append(row)
    return data


def problem_solution(item, index):
    """Accept the aligned release or the original problem/solution schema."""
    problem = item.get("problem", item.get("Question"))
    solution = item.get("L1_full_solution", item.get("solution", item.get("COT_Reason")))
    for name, value in (("problem", problem), ("reference solution", solution)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Row {index}: missing or empty {name}")
    return problem, solution


def check_resume(args, data):
    """Reject a different input, compiler, or generation recipe at the same path."""
    digest = hashlib.sha256()
    with open(args.input, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    signature = {
        "source_sha256": digest.hexdigest(), "rows": len(data),
        "model": args.model or os.environ.get("VLLM_MODEL", "qwen3-8b"),
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_tokens": args.max_tokens, "max_solution_chars": args.max_solution_chars,
        "enable_thinking": args.enable_thinking,
        "prompt_sha256": hashlib.sha256((PROMPT_L4 + PROMPT_L3 + PROMPT_L2).encode()).hexdigest(),
    }
    metadata = Path(args.output + ".meta.json")
    existing = load_jsonl(args.output) if os.path.exists(args.output) else []
    if existing and not metadata.exists():
        raise ValueError("Existing output has no generation metadata; choose a new output path")
    if metadata.exists() and json.loads(metadata.read_text()) != signature:
        raise ValueError("Output metadata does not match this input/compiler/recipe; choose a new output path")
    if len(existing) > len(data):
        raise ValueError("Existing output is longer than the input")
    for index, (source, generated) in enumerate(zip(data, existing)):
        if problem_solution(source, index) != problem_solution(generated, index):
            raise ValueError(f"Resume row {index}: source content/order mismatch")
        hints = generated.get("hints_abstraction", {})
        for key in ("L2_strategy", "L3_meta", "L4_classification"):
            value = hints.get(key)
            if not isinstance(value, str) or not value.strip() or value.lstrip().startswith("[ERROR"):
                raise ValueError(f"Resume row {index}: invalid {key}; repair or restart this output")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if not metadata.exists():
        metadata.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    return len(existing)


def clean_hint(text):
    """Strip common noise: leading/trailing whitespace, surrounding quotes, markdown fences."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
        t = t[1:-1].strip()
    return t


def approx_tokens(text):
    """Rough token count via whitespace split × 1.3 (matches typical BPE ratio for English)."""
    if not text:
        return 0
    return int(round(len(text.split()) * 1.3))


def generate_with_retry(client, prompt, max_retries=3, label=""):
    """
    Call the LLM once and clean the output.
    Retries only on exceptions or empty output — NOT on length (per v3.1 design).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            raw = client.chat(messages)
            cleaned = clean_hint(raw)
            if not cleaned:
                raise ValueError(f"Empty response for {label}")
            if "<think>" in cleaned.lower() or "</think>" in cleaned.lower():
                raise ValueError(f"Thinking trace leaked into final response for {label}")
            return cleaned
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                continue
    raise last_exc


def generate_staged(client, problem, solution, max_retries=3):
    """
    Run the staged L4 → L3 → L2 pipeline for one sample.

    Returns a dict:
      {"L4_classification": str, "L3_meta": str, "L2_strategy": str}
    """
    # ── Stage 1: L4 (independent) ──
    prompt_l4 = PROMPT_L4.format(problem=problem, solution=solution)
    L4 = generate_with_retry(client, prompt_l4, max_retries=max_retries, label="L4")

    # ── Stage 2: L3 (sees L4) ──
    prompt_l3 = PROMPT_L3.format(problem=problem, solution=solution, L4=L4)
    L3 = generate_with_retry(client, prompt_l3, max_retries=max_retries, label="L3")

    # ── Stage 3: L2 (sees L4 + L3) ──
    prompt_l2 = PROMPT_L2.format(problem=problem, solution=solution, L4=L4, L3=L3)
    L2 = generate_with_retry(client, prompt_l2, max_retries=max_retries, label="L2")

    return {
        "L4_classification": L4,
        "L3_meta": L3,
        "L2_strategy": L2,
    }


def summarize_length_distribution(output_path):
    """
    Post-hoc: scan the output JSONL and print a length distribution summary
    per level. This distribution is itself part of the research findings
    (natural information density of each abstraction level).
    """
    if not os.path.exists(output_path):
        return

    lens = {"L4": [], "L3": [], "L2": [], "L1_solution": []}
    n_items = 0
    n_errors = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_items += 1
            ha = item.get("hints_abstraction", {})
            for key, level_name in [
                ("L4_classification", "L4"),
                ("L3_meta", "L3"),
                ("L2_strategy", "L2"),
            ]:
                v = ha.get(key, "")
                if isinstance(v, str) and not v.startswith("[ERROR"):
                    lens[level_name].append(approx_tokens(v))
                elif isinstance(v, str) and v.startswith("[ERROR"):
                    n_errors += 1
            sol = item.get("solution", "")
            if isinstance(sol, str) and sol:
                lens["L1_solution"].append(approx_tokens(sol))

    print()
    print("═" * 74)
    print("  Length Distribution Summary (approx. tokens per hint)")
    print("═" * 74)
    print(f"  Total items scanned: {n_items}")
    print(f"  Level                n     mean   median   std     p10    p90")
    print(f"  " + "-" * 68)
    for level_name in ["L1_solution", "L2", "L3", "L4"]:
        vals = lens[level_name]
        if not vals:
            continue
        n = len(vals)
        m = mean(vals)
        med = median(vals)
        s = stdev(vals) if n > 1 else 0.0
        vs = sorted(vals)
        p10 = vs[max(0, int(0.10 * (n - 1)))]
        p90 = vs[min(n - 1, int(0.90 * (n - 1)))]
        print(f"  {level_name:<20} {n:<5} {m:>6.1f}  {med:>5.1f}  {s:>6.1f}  {p10:>5}  {p90:>5}")
    print("═" * 74)
    print("  Note: L1 length is the raw `solution` field (pre-truncation) —")
    print("  it is naturally 10-50x longer than L2-L4 by design, and this")
    print("  ratio is itself a data point about the natural information")
    print("  density of each abstraction level.")
    print("═" * 74)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate L2/L3/L4 abstraction-level hints via staged LLM calls "
                    "with natural (unconstrained) length (v3.1)"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--base-url", default=None, help="vLLM/OpenAI base URL")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max output tokens per LLM call (upper bound only; "
                             "the LLM chooses its natural length within this cap)")
    parser.add_argument("--max-solution-chars", type=int, default=4000,
                        help="Truncate solution to this many chars to avoid context overflow")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen thinking mode; only the final content is saved")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retries per level on exception (NOT on length)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of samples generated concurrently; each sample "
                             "still follows L4 -> L3 -> L2 sequentially")
    parser.add_argument("--fail-on-error", action="store_true",
                        help="Stop before writing a failed row, preserving exact "
                             "line-count resume semantics")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to first N samples (for testing)")
    parser.add_argument("--skip-length-summary", action="store_true",
                        help="Skip the post-run length distribution summary")
    args = parser.parse_args()

    if args.concurrency < 1:
        raise ValueError(f"--concurrency must be >= 1, got {args.concurrency}")
    if args.max_solution_chars < 1:
        raise ValueError("--max-solution-chars must be positive")

    # Validate every reference before constructing a client or making API calls.
    data = load_jsonl(args.input)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        data = data[:args.max_samples]
    if not data:
        raise ValueError("Input dataset is empty")
    sources = [problem_solution(item, index) for index, item in enumerate(data)]
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output must be different files")
    done_count = check_resume(args, data)

    client = LLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # ── Load input samples ──
    total = len(data)
    print(f"[v3.1-natural] Loaded {total} samples from {args.input}")

    # ── Auto-resume ──
    if done_count == total:
        print(f"[v3.1-natural] Output already complete ({done_count}/{total}), nothing to do.")
        if not args.skip_length_summary:
            summarize_length_distribution(args.output)
        return
    if done_count > 0:
        print(f"[v3.1-natural] Resuming from sample {done_count}/{total} "
              f"({done_count} already in {args.output})")

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ── Streaming generation ──
    success_count = 0
    error_count = 0
    stage_failure_stats = {"L4": 0, "L3": 0, "L2": 0}

    def generate_index(index):
        """Generate one row without mutating shared input state."""
        problem, solution = sources[index]
        if len(solution) > args.max_solution_chars:
            solution = solution[:args.max_solution_chars] + "\n... (solution truncated)"
        try:
            hints = generate_staged(
                client, problem, solution, max_retries=args.max_retries
            )
            return index, hints, None
        except Exception as exc:
            return index, None, str(exc)

    print(
        f"[v3.1-natural] Generating with concurrency={args.concurrency}; "
        f"fail_on_error={args.fail_on_error}"
    )

    # Keep the vLLM server continuously batched, but commit rows strictly in
    # input order. Therefore an interrupted run can resume exactly from the
    # number of complete JSONL rows already on disk.
    with open(args.output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            in_flight = {}
            completed = {}
            next_submit = done_count
            next_write = done_count

            while next_submit < total and len(in_flight) < args.concurrency:
                future = executor.submit(generate_index, next_submit)
                in_flight[future] = next_submit
                next_submit += 1

            with tqdm(
                desc="[v3.1-natural] L4→L3→L2",
                initial=done_count,
                total=total,
            ) as pbar:
                while in_flight:
                    finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in finished:
                        expected_index = in_flight.pop(future)
                        index, hints, err_msg = future.result()
                        if index != expected_index:
                            raise RuntimeError(
                                f"worker index mismatch: expected {expected_index}, got {index}"
                            )
                        completed[index] = (hints, err_msg)

                    while next_write in completed:
                        hints, err_msg = completed.pop(next_write)
                        if err_msg is not None and args.fail_on_error:
                            raise RuntimeError(
                                f"Sample {next_write} failed after retries; "
                                f"output remains resumable at this row: {err_msg}"
                            )

                        item = dict(data[next_write])
                        if err_msg is None:
                            item["hints_abstraction"] = hints
                            success_count += 1
                        else:
                            print(
                                f"\n[v3.1-natural][ERROR] Sample {next_write}: {err_msg}"
                            )
                            for stg in ["L4", "L3", "L2"]:
                                if stg in err_msg:
                                    stage_failure_stats[stg] += 1
                                    break
                            item["hints_abstraction"] = {
                                "L4_classification": f"[ERROR: {err_msg}]",
                                "L3_meta": f"[ERROR: {err_msg}]",
                                "L2_strategy": f"[ERROR: {err_msg}]",
                            }
                            error_count += 1

                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        out_f.flush()
                        next_write += 1
                        pbar.update(1)

                    # Do not let a single slow thinking response create an
                    # unbounded out-of-order buffer. Keep at most one extra
                    # concurrency window waiting behind the next row to write.
                    while (
                        next_submit < total
                        and len(in_flight) < args.concurrency
                        and len(completed) < args.concurrency
                    ):
                        new_future = executor.submit(generate_index, next_submit)
                        in_flight[new_future] = next_submit
                        next_submit += 1

                    pbar.set_postfix(
                        ok=success_count,
                        err=error_count,
                        inflight=len(in_flight),
                        buffered=len(completed),
                    )

    print(f"[v3.1-natural] Done: {success_count} success, {error_count} errors, "
          f"total in output: {count_lines(args.output)}/{total}")
    if error_count > 0:
        print(f"[v3.1-natural] Stage failure breakdown: {stage_failure_stats}")

    # ── Post-run length distribution summary ──
    if not args.skip_length_summary:
        summarize_length_distribution(args.output)


if __name__ == "__main__":
    main()
