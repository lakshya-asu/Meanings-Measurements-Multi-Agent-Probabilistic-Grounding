# Dataset CSV audit (gate 2, frozen 2026-08-15)

Audit of every CSV under `datasets/explore-eqa/` and `datasets/_drive_root_csvs/`.
Nothing was deleted or moved. Row counts exclude the header. SHA256 values are
truncated to 12 hex chars in the tables; full hashes for the canonical file and
the frozen split are in `splits/MANIFEST.json`.

## Canonical benchmark file

Exactly one file is canonical:

| Path | sha256 (first 12) | Bytes | Rows | Scenes |
|---|---|---|---|---|
| `datasets/explore-eqa/final files/questions_msp_eval_combined_100.csv` | `ae3e2429eb08` | 27549 | 98 | 41 |

Full sha256: `ae3e2429eb08876577f1d871c0108252cc236fa53bf2c6985f1f70f1755e1a8f`

This file is frozen byte-for-byte as `splits/bench_v1_98.csv` and pinned in
`splits/MANIFEST.json`. All runs must load it through `src/splits.py`
(`load_split("bench_v1_98")`), never by path.

Provenance note: the canonical 98 rows are exactly the union of
`final files/spatial_questions_first50.csv` (49 rows) and
`final files/spatial_questions_last50.csv` (49 rows). Every row of each half
appears in the canonical file and nothing else does.

## Files in the combined_100 family (near-duplicates of the canonical bench)

| Path | sha256 (first 12) | Bytes | Rows | Identical to canonical? |
|---|---|---|---|---|
| `datasets/explore-eqa/questions_msp_eval_combined_100.csv` | `bb7315c33022` | 27826 | 98 | No. Adds an `id` column. On the shared columns, 0 of 98 rows differ. |
| `datasets/explore-eqa/questions_msp_eval_combined_100 (copy).csv` | `bb7315c33022` | 27826 | 98 | No. Byte-identical to the top-level file above (same extra `id` column, 0 rows differ on shared columns). |
| `datasets/explore-eqa/questions_msp_eval_combined_100_commonsense.csv` | `aa085ab86951` | 28089 | 98 | No. Adds `id` and `Category` columns. On the shared columns, 0 of 98 rows differ. |
| `datasets/explore-eqa/questions_msp_eval_combined_100_ablation_occlusion.csv` | `6665ded0e00f` | 3101 | 10 | No. 10-row occlusion ablation subset with a different header. Not a bench duplicate. |

## " (copy)" files

| Path | sha256 (first 12) | Bytes | Rows | Relation to its non-copy sibling | Identical to canonical? |
|---|---|---|---|---|---|
| `datasets/explore-eqa/questions_msp_eval_combined_100 (copy).csv` | `bb7315c33022` | 27826 | 98 | Byte-identical to `questions_msp_eval_combined_100.csv` in the same dir | No (see family table above) |
| `datasets/explore-eqa/questions_msp_sample_1 (copy).csv` | `7ab1ba60f456` | 9755 | 50 | Differs from sibling (`73fc09968362`, 49 rows): different row count and content | No (different schema) |
| `datasets/explore-eqa/scene_init_poses_semantic_only (copy).csv` | `6b4593016d72` | 12453 | 191 | Differs from sibling (`0faa6ad7e290`, 30 rows): the copy has 161 more rows | No (pose file, not questions) |
| `datasets/_drive_root_csvs/questions_msp_sample_1 (copy).csv` | `591b0ceb2865` | 9759 | 50 | Differs from sibling (`462000504aeb`, 45 rows): different row count and content | No (different schema) |

Note: the two `questions_msp_sample_1 (copy).csv` files in the two directories
also differ from each other (`7ab1ba60f456` vs `591b0ceb2865`).

## Cross-directory duplicates: explore-eqa vs _drive_root_csvs

Files with the same name in both directories.

| Name | explore-eqa sha256 (first 12) | _drive_root sha256 (first 12) | Byte-identical pair? | Rows (explore-eqa / _drive_root) |
|---|---|---|---|---|
| `questions.csv` | `a7476e727501` | `a7476e727501` | Yes | 500 / 500 |
| `questions_filtered.csv` | `16be91cc76e0` | `16be91cc76e0` | Yes | 114 / 114 |
| `questions_msp.csv` | `ac55ae169ab4` | `ac55ae169ab4` | Yes | 912 / 912 |
| `questions_msp_sample_1.csv` | `73fc09968362` | `462000504aeb` | No | 49 / 45 |
| `questions_msp_sample_1 (copy).csv` | `7ab1ba60f456` | `591b0ceb2865` | No | 50 / 50 |
| `questions_msp_sample_2.csv` | `b0d507f5959b` | `b0d507f5959b` | Yes | 50 / 50 |
| `questions_msp_sample_3.csv` | `e437991bafaa` | `e437991bafaa` | Yes | 50 / 50 |
| `questions_msp_sample_4.csv` | `50037aeb804f` | `50037aeb804f` | Yes | 50 / 50 |
| `scene_init_poses.csv` | `b52a9a389e5a` | `b52a9a389e5a` | Yes | 1015 / 1015 |
| `scene_init_poses_semantic_only.csv` | `0faa6ad7e290` | `66fbccddea1c` | No | 30 / 189 |

None of these are the canonical bench file.

## Duplicates hiding under different names

| File A | File B | Relation |
|---|---|---|
| `datasets/explore-eqa/questions_msp_eval.csv` (`5e2c362098f0`, 49 rows) | `datasets/explore-eqa/final files/spatial_questions_first50.csv` (`5e2c362098f0`, 49 rows) | Byte-identical. All 49 rows are contained in the canonical bench. |
| `datasets/explore-eqa/questions_msp_sample_new.csv` (`fa63f0e6c8f7`, 49 rows) | `datasets/explore-eqa/final files/spatial_questions_last50.csv` (`fa63f0e6c8f7`, 49 rows) | Byte-identical. All 49 rows are contained in the canonical bench. |

## Remaining non-duplicate CSVs (listed for completeness)

| Path | sha256 (first 12) | Bytes | Rows |
|---|---|---|---|
| `datasets/explore-eqa/questions_msp_sample_1_ object.csv` | `017fc68941be` | 12218 | 49 |
| `datasets/explore-eqa/questions_msp_sample_1_metric_corrected.csv` | `cc84769cb496` | 12958 | 44 |
| `datasets/explore-eqa/scene_init_poses_semantic_only_new.csv` | `9774fa9e470f` | 2111 | 49 |
| `datasets/explore-eqa/scene_init_poses_semantic_only_occlusion.csv` | `f6ec6506261c` | 473 | 10 |

## Other observations

1. `datasets/explore-eqa/explore-eqa/final files/` exists and is empty. It looks
   like a stray nesting from a Drive export.
2. Seven of the ten same-named files in `datasets/_drive_root_csvs/` are exact
   byte duplicates of the `datasets/explore-eqa/` versions. The three that
   differ (`questions_msp_sample_1.csv`, its copy, and
   `scene_init_poses_semantic_only.csv`) are older or divergent snapshots.
3. The two near-duplicates of the bench at the top level of
   `datasets/explore-eqa/` (`questions_msp_eval_combined_100.csv` and its
   " (copy)") carry the same 98 questions plus an `id` column. They are safe in
   content today but are exactly the kind of file that drifts. Do not load them.
   Only `splits/bench_v1_98.csv` via the verifying loader is authoritative.
