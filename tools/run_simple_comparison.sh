#!/bin/bash
# Simple script to compare SGLang and FSDP tensor dumps
#
# Usage:
#   1. Run training with debug_one_sample mode:
#      cd /root/slime && python examples/true_on_policy/run_simple_dumper.py
#
#   2. After training completes, run this comparison:
#      bash /root/slime/tools/run_simple_comparison.sh

set -e

SGLANG_DIR="${SGLANG_DIR:-/tmp/sglang_tensor_dump}"
FSDP_DIR="${FSDP_DIR:-/tmp/fsdp_tensor_dump}"

echo "================================"
echo "SGLang vs FSDP Tensor Comparison"
echo "================================"
echo ""
echo "SGLang dump dir: ${SGLANG_DIR}"
echo "FSDP dump dir:   ${FSDP_DIR}"
echo ""

# Check if directories exist
if [ ! -d "${SGLANG_DIR}" ]; then
    echo "ERROR: SGLang dump directory not found: ${SGLANG_DIR}"
    echo "Make sure you ran the training with SGLANG_DUMPER_ENABLE=1"
    exit 1
fi

if [ ! -d "${FSDP_DIR}" ]; then
    echo "ERROR: FSDP dump directory not found: ${FSDP_DIR}"
    echo "Make sure you ran the training with FSDP_TENSOR_DUMP_DIR set"
    exit 1
fi

# List available passes
echo "Available SGLang passes:"
find "${SGLANG_DIR}" -name "Pass*.pt" | head -20
echo ""

echo "Available FSDP passes:"
find "${FSDP_DIR}" -name "Pass*.pt" | head -20
echo ""

# Run comparison
echo "Running comparison..."
python /root/slime/tools/simple_compare.py \
    --sglang-dir "${SGLANG_DIR}" \
    --fsdp-dir "${FSDP_DIR}" \
    --fsdp-pass 0 \
    --temperature 0.8

