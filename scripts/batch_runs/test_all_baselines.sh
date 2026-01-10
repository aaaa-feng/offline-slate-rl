#!/bin/bash
################################################################################
# Phase 3 Final Acceptance Test: All-Algorithm Smoke Test
#
# Purpose: Validate all 4 offline RL baselines with dual-stream E2E GRU
# Environment: diffuse_mix
# Dataset: medium
# Scale: 10k steps (quick validation)
#
# Author: Architect
# Date: 2026-01-07
################################################################################

set -e  # Exit on error
set -o pipefail  # Ensure pipeline failures are caught

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT="/data/liyuefeng/offline-slate-rl"
LOG_DIR="${PROJECT_ROOT}/experiments/logs/test/offline_test_20260107"
SRC_DIR="${PROJECT_ROOT}/src"
PYTHON="/data/liyuefeng/miniconda3/envs/gems/bin/python"

# Test parameters
ENV_NAME="diffuse_mix"
DATASET_QUALITY="medium"
MAX_TIMESTEPS=10000
EVAL_FREQ=2000
SAVE_FREQ=5000
EXPERIMENT_NAME="smoke_test_v1"
SEED=58407201
DEVICE="cuda"

# ============================================================================
# Setup
# ============================================================================

echo "================================================================================"
echo "Phase 3 Final Acceptance Test: All-Algorithm Smoke Test"
echo "================================================================================"
echo "Environment: ${ENV_NAME}"
echo "Dataset: ${DATASET_QUALITY}"
echo "Max Timesteps: ${MAX_TIMESTEPS}"
echo "Log Directory: ${LOG_DIR}"
echo "================================================================================"
echo ""

# Create log directory
mkdir -p "${LOG_DIR}"

# Change to project root
cd "${PROJECT_ROOT}"

# ============================================================================
# Test 1: BC (Behavior Cloning)
# ============================================================================

echo "[1/4] Testing BC (Behavior Cloning)..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"

${PYTHON} -u "${SRC_DIR}/agents/offline/bc.py" \
    --env_name "${ENV_NAME}" \
    --dataset_quality "${DATASET_QUALITY}" \
    --max_timesteps ${MAX_TIMESTEPS} \
    --eval_freq ${EVAL_FREQ} \
    --save_freq ${SAVE_FREQ} \
    --experiment_name "${EXPERIMENT_NAME}" \
    --seed ${SEED} \
    --device "${DEVICE}" \
    --no_swanlab \
    2>&1 | tee "${LOG_DIR}/bc_smoke_test.log"

BC_EXIT_CODE=$?
echo "BC Exit Code: ${BC_EXIT_CODE}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${BC_EXIT_CODE} -ne 0 ]; then
    echo "❌ BC test FAILED with exit code ${BC_EXIT_CODE}"
    exit 1
fi

echo "✅ BC test PASSED"
echo ""

# ============================================================================
# Test 2: TD3+BC
# ============================================================================

echo "[2/4] Testing TD3+BC..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"

${PYTHON} -u "${SRC_DIR}/agents/offline/td3_bc.py" \
    --env_name "${ENV_NAME}" \
    --dataset_quality "${DATASET_QUALITY}" \
    --max_timesteps ${MAX_TIMESTEPS} \
    --eval_freq ${EVAL_FREQ} \
    --save_freq ${SAVE_FREQ} \
    --experiment_name "${EXPERIMENT_NAME}" \
    --seed ${SEED} \
    --device "${DEVICE}" \
    --alpha 2.5 \
    --no_swanlab \
    2>&1 | tee "${LOG_DIR}/td3bc_smoke_test.log"

TD3BC_EXIT_CODE=$?
echo "TD3+BC Exit Code: ${TD3BC_EXIT_CODE}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${TD3BC_EXIT_CODE} -ne 0 ]; then
    echo "❌ TD3+BC test FAILED with exit code ${TD3BC_EXIT_CODE}"
    exit 1
fi

echo "✅ TD3+BC test PASSED"
echo ""

# ============================================================================
# Test 3: CQL (Conservative Q-Learning)
# ============================================================================

echo "[3/4] Testing CQL (Conservative Q-Learning)..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"

${PYTHON} -u "${SRC_DIR}/agents/offline/cql.py" \
    --env_name "${ENV_NAME}" \
    --dataset_quality "${DATASET_QUALITY}" \
    --max_timesteps ${MAX_TIMESTEPS} \
    --eval_freq ${EVAL_FREQ} \
    --save_freq ${SAVE_FREQ} \
    --experiment_name "${EXPERIMENT_NAME}" \
    --seed ${SEED} \
    --device "${DEVICE}" \
    --alpha 5.0 \
    --no_swanlab \
    2>&1 | tee "${LOG_DIR}/cql_smoke_test.log"

CQL_EXIT_CODE=$?
echo "CQL Exit Code: ${CQL_EXIT_CODE}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${CQL_EXIT_CODE} -ne 0 ]; then
    echo "❌ CQL test FAILED with exit code ${CQL_EXIT_CODE}"
    exit 1
fi

echo "✅ CQL test PASSED"
echo ""

# ============================================================================
# Test 4: IQL (Implicit Q-Learning)
# ============================================================================

echo "[4/4] Testing IQL (Implicit Q-Learning)..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"

${PYTHON} -u "${SRC_DIR}/agents/offline/iql.py" \
    --env_name "${ENV_NAME}" \
    --dataset_quality "${DATASET_QUALITY}" \
    --max_timesteps ${MAX_TIMESTEPS} \
    --eval_freq ${EVAL_FREQ} \
    --save_freq ${SAVE_FREQ} \
    --experiment_name "${EXPERIMENT_NAME}" \
    --seed ${SEED} \
    --device "${DEVICE}" \
    --expectile 0.7 \
    --beta 3.0 \
    --no_swanlab \
    2>&1 | tee "${LOG_DIR}/iql_smoke_test.log"

IQL_EXIT_CODE=$?
echo "IQL Exit Code: ${IQL_EXIT_CODE}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${IQL_EXIT_CODE} -ne 0 ]; then
    echo "❌ IQL test FAILED with exit code ${IQL_EXIT_CODE}"
    exit 1
fi

echo "✅ IQL test PASSED"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "================================================================================"
echo "All-Algorithm Smoke Test Summary"
echo "================================================================================"
echo "✅ BC:     PASSED (Exit Code: ${BC_EXIT_CODE})"
echo "✅ TD3+BC: PASSED (Exit Code: ${TD3BC_EXIT_CODE})"
echo "✅ CQL:    PASSED (Exit Code: ${CQL_EXIT_CODE})"
echo "✅ IQL:    PASSED (Exit Code: ${IQL_EXIT_CODE})"
echo "================================================================================"
echo ""
echo "Test logs saved to: ${LOG_DIR}"
echo "  - bc_smoke_test.log"
echo "  - td3bc_smoke_test.log"
echo "  - cql_smoke_test.log"
echo "  - iql_smoke_test.log"
echo ""
echo "🎉 Phase 3 Final Acceptance Test: ALL TESTS PASSED!"
echo "================================================================================"

exit 0
