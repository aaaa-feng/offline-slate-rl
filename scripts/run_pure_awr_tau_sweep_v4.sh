#!/bin/bash
# Pure AWR + High Expectile (Tau) Sweep - V4 GPU隔离版
# 9个实验：3 tau × 3 seeds
# GPU策略：使用CUDA_VISIBLE_DEVICES强制GPU隔离，防止所有进程挤到GPU 0
# 修复：每个进程只能看到并使用它被分配的GPU

PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python
IQL_SCRIPT=/data/liyuefeng/offline-slate-rl/src/agents/offline/iql.py
LOG_DIR=/data/liyuefeng/offline-slate-rl/motivation_test/logs/pure_awr_tau_sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p $LOG_DIR

echo "=========================================="
echo "Pure AWR + Tau Sweep Experiments (V4)"
echo "🔥 使用CUDA_VISIBLE_DEVICES强制GPU隔离"
echo "=========================================="
echo "开始时间: $(date)"
echo "日志目录: $LOG_DIR"
echo ""

# ============================================================
# 实验组1: tau=0.7 (3个seeds) - GPU 2, 3, 4
# ============================================================
echo "启动 tau=0.7 实验组..."

# Seed 42 - GPU 2 (CUDA_VISIBLE_DEVICES=2 → 进程内看到的是cuda:0)
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.7 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau07_seed42 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 42 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.7_seed42_${TIMESTAMP}.log 2>&1 &
echo "  [1/9] tau=0.7, seed=42, GPU=2 (CUDA_VISIBLE_DEVICES=2), PID=$!"

sleep 5

# Seed 12345 - GPU 3
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.7 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau07_seed12345 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 12345 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.7_seed12345_${TIMESTAMP}.log 2>&1 &
echo "  [2/9] tau=0.7, seed=12345, GPU=3 (CUDA_VISIBLE_DEVICES=3), PID=$!"

sleep 5

# Seed 58407201 - GPU 4
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.7 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau07_seed58407201 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 58407201 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.7_seed58407201_${TIMESTAMP}.log 2>&1 &
echo "  [3/9] tau=0.7, seed=58407201, GPU=4 (CUDA_VISIBLE_DEVICES=4), PID=$!"

sleep 5

# ============================================================
# 实验组2: tau=0.8 (3个seeds) - GPU 5, 6, 7
# ============================================================
echo ""
echo "启动 tau=0.8 实验组..."

# Seed 42 - GPU 5
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.8 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau08_seed42 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 42 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.8_seed42_${TIMESTAMP}.log 2>&1 &
echo "  [4/9] tau=0.8, seed=42, GPU=5 (CUDA_VISIBLE_DEVICES=5), PID=$!"

sleep 5

# Seed 12345 - GPU 6
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.8 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau08_seed12345 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 12345 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.8_seed12345_${TIMESTAMP}.log 2>&1 &
echo "  [5/9] tau=0.8, seed=12345, GPU=6 (CUDA_VISIBLE_DEVICES=6), PID=$!"

sleep 5

# Seed 58407201 - GPU 7
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.8 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau08_seed58407201 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 58407201 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.8_seed58407201_${TIMESTAMP}.log 2>&1 &
echo "  [6/9] tau=0.8, seed=58407201, GPU=7 (CUDA_VISIBLE_DEVICES=7), PID=$!"

sleep 5

# ============================================================
# 实验组3: tau=0.9 (3个seeds) - GPU 2, 3, 4 (复用)
# ============================================================
echo ""
echo "启动 tau=0.9 实验组..."

# Seed 42 - GPU 2
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.9 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau09_seed42 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 42 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.9_seed42_${TIMESTAMP}.log 2>&1 &
echo "  [7/9] tau=0.9, seed=42, GPU=2 (CUDA_VISIBLE_DEVICES=2), PID=$!"

sleep 5

# Seed 12345 - GPU 3
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.9 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau09_seed12345 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 12345 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.9_seed12345_${TIMESTAMP}.log 2>&1 &
echo "  [8/9] tau=0.9, seed=12345, GPU=3 (CUDA_VISIBLE_DEVICES=3), PID=$!"

sleep 5

# Seed 58407201 - GPU 4
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.9 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau09_seed58407201 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 58407201 --device cuda:0 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.9_seed58407201_${TIMESTAMP}.log 2>&1 &
echo "  [9/9] tau=0.9, seed=58407201, GPU=4 (CUDA_VISIBLE_DEVICES=4), PID=$!"

echo ""
echo "=========================================="
echo "所有实验已启动！"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "🔥 关键修复："
echo "  - 使用CUDA_VISIBLE_DEVICES强制GPU隔离"
echo "  - 每个进程只能看到1个GPU（进程内使用cuda:0）"
echo "  - GPU 2,3,4各运行2个实验（tau=0.7和tau=0.9）"
echo "  - GPU 5,6,7各运行1个实验（tau=0.8）"
echo ""
echo "查看运行状态："
echo "  ps aux | grep iql.py | grep pure_awr_tau | grep -v grep"
echo ""
echo "查看GPU使用："
echo "  gpustat"
echo ""
echo "查看日志："
echo "  tail -f $LOG_DIR/iql_pure_awr_tau0.7_seed42_${TIMESTAMP}.log"
