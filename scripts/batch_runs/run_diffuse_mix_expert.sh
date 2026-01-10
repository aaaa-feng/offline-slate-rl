#!/bin/bash
################################################################################
# Batch Run Script for Offline RL Algorithms
# Environment: diffuse_mix
# Dataset: expert
# GPUs: 1, 2, 3, 6
################################################################################

set -e  # Exit on error

# Project root directory
PROJECT_ROOT="/data/liyuefeng/offline-slate-rl"
cd "$PROJECT_ROOT"

# Experiment configuration
ENV_NAME="diffuse_mix"
DATASET_QUALITY="expert"
MAX_TIMESTEPS=1000000
SEED=58407201
TIMESTAMP=$(date +"%m%d_%H%M")

echo "================================================================================"
echo "Batch Run Started at $(date)"
echo "================================================================================"
echo "Environment: $ENV_NAME"
echo "Dataset Quality: $DATASET_QUALITY"
echo "Max Timesteps: $MAX_TIMESTEPS"
echo "Seed: $SEED"
echo "Run ID: $TIMESTAMP"
echo "GPUs: 0 (BC+IQL), 1 (TD3+BC), 2 (CQL)"
echo "================================================================================"
echo ""

# Create debug logs directory
mkdir -p logs/.debug

# Function to run algorithm on specific GPU
run_algorithm() {
    local algo_name=$1
    local gpu_id=$2
    local script_path=$3

    echo "[$(date)] Starting $algo_name on GPU $gpu_id..."

    CUDA_VISIBLE_DEVICES=$gpu_id python "$script_path" \
        --max_timesteps $MAX_TIMESTEPS \
        --seed $SEED \
        --run_id $TIMESTAMP \
        > logs/.debug/${algo_name}_${TIMESTAMP}.out 2>&1 &

    local pid=$!
    echo "[$(date)] $algo_name (PID: $pid) launched on GPU $gpu_id"
}

################################################################################
# Launch all 4 algorithms on different GPUs
################################################################################

echo "Launching algorithms..."
echo ""

# 1. BC on GPU 0
run_algorithm "BC" 0 "src/agents/offline/bc.py"
sleep 5  # Wait 5 seconds between launches

# 2. IQL on GPU 0
run_algorithm "IQL" 0 "src/agents/offline/iql.py"
sleep 5

# 3. TD3+BC on GPU 1
run_algorithm "TD3BC" 1 "src/agents/offline/td3_bc.py"
sleep 5

# 4. CQL on GPU 2
run_algorithm "CQL" 2 "src/agents/offline/cql.py"

echo ""
echo "================================================================================"
echo "All algorithms launched successfully!"
echo "================================================================================"
echo ""

# Display running processes
echo "Running processes:"
ps aux | grep "python src/agents/offline" | grep -v grep


echo ""
echo "================================================================================"
echo "Monitoring Information"
echo "================================================================================"
echo ""
echo "Algorithm logs and models:"
echo "  BC:     experiments/logs/offline/log_${SEED}/BC/"
echo "  IQL:    experiments/logs/offline/log_${SEED}/IQL/"
echo "  TD3BC:  experiments/logs/offline/log_${SEED}/TD3_BC/"
echo "  CQL:    experiments/logs/offline/log_${SEED}/CQL/"
echo "================================================================================"
echo "Useful Commands"
echo "================================================================================"
echo ""
echo "# Check running processes:"
echo "ps aux | grep 'python src/agents/offline' | grep -v grep"
echo ""
echo "# Monitor GPU usage:"
echo "watch -n 1 nvidia-smi"
echo ""
echo "# Kill all running experiments:"
echo "pkill -f 'python src/agents/offline'"
echo ""
echo "================================================================================"
echo "Batch run script completed at $(date)"
echo "================================================================================"
