#!/bin/bash
# OneVL inference on Roadwork test set
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# ---- Configurable ----
DECODER_EXPLAIN=${DECODER_EXPLAIN:-false}
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-false}
OUTPUT_PATH=${OUTPUT_PATH:-"${ROOT_DIR}/output/roadwork/roadwork_results.json"}

# ---- Fixed ----
MODEL_PATH=/e2e-data/evad-tech-vla/lujinghui/ms-swift/outputs/roadwork/qwen3_vl_latent_cot_stage2_vis4_txt2_fixbug_512_bs64_with_viscondition/v0-20260324-031631/checkpoint-1260
TEST_SET_PATH=${ROOT_DIR}/test_data/roadwork_test.json
IMAGE_BASE_PATH=/e2e-data/embodied-research-data/opendata/
PREFIX_K=5

export MODEL_PATH TEST_SET_PATH IMAGE_BASE_PATH OUTPUT_PATH PREFIX_K
export DECODER_EXPLAIN VISUAL_DECODER_EXPLAIN

echo "=== Roadwork Inference ==="
echo "  DECODER_EXPLAIN:        ${DECODER_EXPLAIN}"
echo "  VISUAL_DECODER_EXPLAIN: ${VISUAL_DECODER_EXPLAIN}"
echo "  OUTPUT_PATH:            ${OUTPUT_PATH}"
exec bash "${ROOT_DIR}/run_infer.sh"
