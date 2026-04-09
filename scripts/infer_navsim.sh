#!/bin/bash
# OneVL inference on NAVSIM test set
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# ---- Configurable ----
DECODER_EXPLAIN=${DECODER_EXPLAIN:-false}
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-false}
OUTPUT_PATH=${OUTPUT_PATH:-"${ROOT_DIR}/output/navsim/navsim_results.json"}

# ---- Fixed ----
MODEL_PATH=/e2e-data/evad-tech-vla/lujinghui/ms-swift/outputs/navsim/qwen3_vl_latent_cot_stage2_vis4_txt2_fixbug_512_bs64_with_viscondition/v0-20260324-044424/checkpoint-6000
TEST_SET_PATH=${ROOT_DIR}/test_data/navsim_test.json
IMAGE_BASE_PATH=/e2e-data/evad-osc-datasets/datasets/
ANSWER_PREFIX="["

export MODEL_PATH TEST_SET_PATH IMAGE_BASE_PATH OUTPUT_PATH ANSWER_PREFIX
export DECODER_EXPLAIN VISUAL_DECODER_EXPLAIN

echo "=== NAVSIM Inference ==="
echo "  DECODER_EXPLAIN:        ${DECODER_EXPLAIN}"
echo "  VISUAL_DECODER_EXPLAIN: ${VISUAL_DECODER_EXPLAIN}"
echo "  OUTPUT_PATH:            ${OUTPUT_PATH}"
exec bash "${ROOT_DIR}/run_infer.sh"
