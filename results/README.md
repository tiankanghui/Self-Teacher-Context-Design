# Compact recorded results

- `id_summary/id_run_summary.csv`: unrounded ID scores for each available run
  under the four checkpoint-reporting rules.
- `id_summary/id_paired_seed_summary.csv`: paired L3/L4 differences from L1
  across seeds 42, 43 and 44.
- `step200_scores.csv`: aggregate ID and transfer scores at step 200, with row
  counts and generation counts.
- `exact_token_summary.csv`: exact Qwen3 token statistics for L1--L5.

These tables are included for quick verification of the main empirical claims.
Per-item outputs, model-judge labels and diagnostic geometry results belong to
the separate full artifact and are omitted here.
