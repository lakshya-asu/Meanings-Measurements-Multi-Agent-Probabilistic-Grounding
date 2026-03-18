#!/usr/bin/env bash
set -euo pipefail

CSV="/datasets/explore-eqa/questions_msp_sample_1_metric_corrected.csv"

# Example MSP outputs you showed (adjust these to your actual inside-container paths)
# If your jsons are on the host, they must be bind-mounted into the container to exist here.
MSP_POINT_JSON="/root/graph_eqa/src/outputs_benchmark_point/mapg_spatial_gemini_habitat_benchmark/gemini_images_True.json"
MSP_OBJECT_JSON="/root/graph_eqa/src/outputs_benchmark_object/mapg_spatial_gemini_habitat_benchmark/gemini_images_True.json"

OUT_DIR="/root/graph_eqa/src/evals/out"
mkdir -p "$OUT_DIR"

python eval_offset_distances.py \
  --csv "$CSV" \
  --json "$MSP_POINT_JSON" \
  --out "$OUT_DIR/eval_msp_point.csv" \
  --method "msp_point"

python eval_offset_distances.py \
  --csv "$CSV" \
  --json "$MSP_OBJECT_JSON" \
  --out "$OUT_DIR/eval_msp_object.csv" \
  --method "msp_object"

echo "[OK] wrote evals into: $OUT_DIR"
ls -lh "$OUT_DIR"