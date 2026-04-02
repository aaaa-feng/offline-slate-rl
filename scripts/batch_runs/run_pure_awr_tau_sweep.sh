#!/bin/bash
################################################################################
# Pure AWR + High Expectile 战略实验
# 目标：通过提高τ强行拉开Q-V差距，激活AWR的精英筛选能力
#
# 实验假说：
#   - τ=0.5时，V≈Q的中位数，导致Advantage≈0
#   - τ=0.7/0.8/0.9时，V逼近上分位数，强行拉开Q-V差距
#   - 在Pure AWR (λ_BC=0.0) 下，高τ的精英梯度将彻底释放
#
# 实验配置：
#   - 基础：λ_BC=0.0 (Pure AWR), β=3.0, Ranker=GeMS
#   - 扫参：τ ∈ {0.7, 0.8, 0.9}
#   - 种子：{42, 12345, 58407201}
#   - 总计：9个实验 (3个τ × 3个seed)
#   - GPU分配：轮流使用 cuda:2, cuda:3, cuda:5
################################################################################

# Python解释器路径
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

# 项目路径
PROJECT_ROOT=/data/liyuefeng/offline-slate-rl
IQL_SCRIPT=$PROJECT_ROOT/src/agents/offline/iql.py

# 日志目录
LOG_DIR=$PROJECT_ROOT/motivation_test/logs/pure_awr_tau_sweep
mkdir -p $LOG_DIR

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================================"
echo "🎯 Pure AWR + High Expectile 战略实验"
echo "================================================================================================"
echo "实验假说: 提高τ强行拉开Q-V差距，激活AWR的精英筛选能力"
echo "实验配置: λ_BC=0.0 (Pure AWR), β=3.0, τ ∈ {0.7, 0.8, 0.9}"
echo "实验数量: 9个实验 (3个τ × 3个seed)"
echo "日志目录: $LOG_DIR"
echo "================================================================================================"
echo ""

################################################################################
# τ=0.7 实验组 (3个seed)
################################################################################

echo "[1/9] 启动实验: τ=0.7, Seed=42, GPU=cuda:2"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.7 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau07 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 42 --device cuda:2 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.7_seed42_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

echo "[2/9] 启动实验: τ=0.7, Seed=12345, GPU=cuda:3"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.7 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau07 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 12345 --device cuda:3 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.7_seed12345_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

echo "[3/9] 启动实验: τ=0.7, Seed=58407201, GPU=cuda:5"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.7 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau07 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 58407201 --device cuda:5 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.7_seed58407201_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

################################################################################
# τ=0.8 实验组 (3个seed)
################################################################################

echo "[4/9] 启动实验: τ=0.8, Seed=42, GPU=cuda:2"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.8 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau08 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 42 --device cuda:2 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.8_seed42_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

echo "[5/9] 启动实验: τ=0.8, Seed=12345, GPU=cuda:3"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.8 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau08 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 12345 --device cuda:3 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.8_seed12345_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

echo "[6/9] 启动实验: τ=0.8, Seed=58407201, GPU=cuda:5"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.8 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau08 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 58407201 --device cuda:5 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.8_seed58407201_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

################################################################################
# τ=0.9 实验组 (3个seed)
################################################################################

echo "[7/9] 启动实验: τ=0.9, Seed=42, GPU=cuda:2"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.9 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau09 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 42 --device cuda:2 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.9_seed42_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

echo "[8/9] 启动实验: τ=0.9, Seed=12345, GPU=cuda:3"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.9 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau09 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 12345 --device cuda:3 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.9_seed12345_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

echo "[9/9] 启动实验: τ=0.9, Seed=58407201, GPU=cuda:5"
nohup $PYTHON $IQL_SCRIPT \
  --env_name mix_divpen --dataset_quality v2_b5 --ranker gems \
  --expectile 0.9 --beta 3.0 --lambda_bc 0.0 \
  --experiment_tag pure_awr_tau09 \
  --max_timesteps 20000 --eval_freq 500 --log_freq 200 \
  --eval_episodes 50 --final_eval_episodes 100 --best_checkpoint_metric iqm \
  --batch_size 256 --seed 58407201 --device cuda:5 \
  --swan_project "Offline_Slate_RL_202603" \
  > $LOG_DIR/iql_pure_awr_tau0.9_seed58407201_${TIMESTAMP}.log 2>&1 &

echo "  ✓ 进程已启动 (PID: $!)"
sleep 2

################################################################################
# 完成提示
################################################################################

echo ""
echo "================================================================================================"
echo "✅ 所有9个实验已启动完成！"
echo "================================================================================================"
echo ""
echo "📁 日志文件位置:"
echo "  $LOG_DIR/"
echo ""
echo "🔍 监控命令:"
echo "  查看所有进程: ps aux | grep iql.py | grep pure_awr_tau"
echo "  查看日志: tail -f $LOG_DIR/iql_pure_awr_tau*_${TIMESTAMP}.log"
echo "  查看SwanLab: https://swanlab.cn/@Cliff/Offline_Slate_RL_202603"
echo ""
echo "📊 关键监控指标 (13个类别):"
echo "  [5] Advantage: mean, std (是否被拉开？)"
echo "  [7] AWR-Weight: mean, max, std (是否有差异？)"
echo "  [8] Policy: entropy, awr_weight_std (策略是否更探索？)"
echo "  [12] Gradient: actor_norm (梯度是否恢复？)"
echo "  [13] Representation: critic_svd_rank (表征是否稳定？)"
echo "  [Evaluation] IQM Reward (性能是否提升？)"
echo ""
echo "🎯 实验成功判断标准 (Step 15000):"
echo "  τ=0.7: Advantage Mean >0.05, AWR Weight Std >0.10, Peak IQM >140"
echo "  τ=0.8: Advantage Mean >0.10, AWR Weight Std >0.15, Peak IQM >150"
echo "  τ=0.9: Advantage Mean >0.15, AWR Weight Std >0.20, Peak IQM >130"
echo ""
echo "⏱️  预计完成时间: ~2-3小时 (每个实验约15-20分钟)"
echo "================================================================================================"
