#!/bin/bash
# 全面数据收集测试脚本
# 测试所有参数组合和数据收集方式

set -e  # 遇到错误立即退出

# ============ 配置参数 ============
BOREDOM_THRESHOLDS=(3 4)
DIVERSITY_PENALTIES=(2.0 3.0 4.0)
EPISODE_LENGTH=150
NUM_EPISODES=50  # 小规模测试

# 输出目录
OUTPUT_DIR="/data/liyuefeng/offline-slate-rl/results/comprehensive_test"
mkdir -p $OUTPUT_DIR

# 项目根目录
PROJECT_ROOT="/data/liyuefeng/offline-slate-rl"
cd $PROJECT_ROOT

# ============ 函数定义 ============
run_test() {
    local approach=$1
    local env_name=$2
    local quality=$3
    local boredom=$4
    local penalty=$5
    local epsilon=$6
    local test_name=$7

    echo "Running: $test_name"

    python src/data_collection/offline_data_collection/collect_data.py \
        --env_name $env_name \
        --quality $quality \
        --episodes $NUM_EPISODES \
        --boredom_threshold $boredom \
        --diversity_penalty $penalty \
        --episode_length $EPISODE_LENGTH \
        --epsilon_greedy $epsilon \
        --output_dir "$OUTPUT_DIR/$approach/$test_name" \
        > "$OUTPUT_DIR/$approach/${test_name}.log" 2>&1

    echo "✓ Completed: $test_name"
}

# ============ 主测试循环 ============
echo "========================================="
echo "全面数据收集测试开始"
echo "========================================="

for APPROACH in "A" "B"; do
    echo ""
    echo "========================================="
    echo "测试方案 $APPROACH"
    echo "========================================="

    # 设置环境名称（方案A和B使用相同的环境名）
    MIXED_ENV="diffuse_mix"
    TOPDOWN_ENV="diffuse_topdown"

    mkdir -p "$OUTPUT_DIR/$APPROACH"

    # ============ 测试1: Expert-Hard (所有参数组合) ============
    echo ""
    echo "--- 测试 Expert-Hard (6种参数组合) ---"

    for ENV in "$MIXED_ENV" "$TOPDOWN_ENV"; do
        for BOREDOM in "${BOREDOM_THRESHOLDS[@]}"; do
            for PENALTY in "${DIVERSITY_PENALTIES[@]}"; do
                TEST_NAME="${ENV}_expert_hard_b${BOREDOM}_p${PENALTY}"
                run_test "$APPROACH" "$ENV" "expert" "$BOREDOM" "$PENALTY" "0.0" "$TEST_NAME"
            done
        done
    done

    # ============ 测试2: Expert-ε-greedy (原始环境参数) ============
    echo ""
    echo "--- 测试 Expert-ε-greedy (原始参数 + ε=0.3) ---"

    for ENV in "$MIXED_ENV" "$TOPDOWN_ENV"; do
        # 使用原始环境参数（不覆盖）
        TEST_NAME="${ENV}_expert_epsilon"

        python src/data_collection/offline_data_collection/collect_data.py \
            --env_name $ENV \
            --quality expert \
            --episodes $NUM_EPISODES \
            --epsilon_greedy 0.3 \
            --epsilon_noise_scale 1.0 \
            --output_dir "$OUTPUT_DIR/$APPROACH/$TEST_NAME" \
            > "$OUTPUT_DIR/$APPROACH/${TEST_NAME}.log" 2>&1

        echo "✓ Completed: $TEST_NAME"
    done

    # ============ 测试3: Random-Uniform (原始环境参数) ============
    echo ""
    echo "--- 测试 Random-Uniform (原始参数) ---"

    for ENV in "$MIXED_ENV" "$TOPDOWN_ENV"; do
        TEST_NAME="${ENV}_random_uniform"

        python src/data_collection/offline_data_collection/collect_data.py \
            --env_name $ENV \
            --quality random \
            --episodes $NUM_EPISODES \
            --output_dir "$OUTPUT_DIR/$APPROACH/$TEST_NAME" \
            > "$OUTPUT_DIR/$APPROACH/${TEST_NAME}.log" 2>&1

        echo "✓ Completed: $TEST_NAME"
    done

    # ============ 测试4: Random-Hard (3种代表性参数) ============
    echo ""
    echo "--- 测试 Random-Hard (3种代表性参数) ---"

    # 选择3种代表性参数组合
    REPRESENTATIVE_PARAMS=(
        "3 3.0"   # 中等严格
        "3 4.0"   # 较严格
        "4 3.0"   # 较宽松
    )

    for ENV in "$MIXED_ENV" "$TOPDOWN_ENV"; do
        for PARAMS in "${REPRESENTATIVE_PARAMS[@]}"; do
            read BOREDOM PENALTY <<< "$PARAMS"
            TEST_NAME="${ENV}_random_hard_b${BOREDOM}_p${PENALTY}"
            run_test "$APPROACH" "$ENV" "random" "$BOREDOM" "$PENALTY" "0.0" "$TEST_NAME"
        done
    done

done

echo ""
echo "========================================="
echo "全面数据收集测试完成！"
echo "========================================="
echo "结果保存在: $OUTPUT_DIR"
