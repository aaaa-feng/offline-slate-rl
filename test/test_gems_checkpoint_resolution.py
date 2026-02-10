"""
测试GeMS checkpoint路径解析逻辑
验证epsilon-greedy模型是否能被正确识别和加载
"""
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.offline.checkpoint_utils import resolve_gems_checkpoint

def test_gems_checkpoint_resolution():
    """测试不同dataset_quality参数的GeMS路径解析"""

    test_cases = [
        {
            "name": "v2_b3 (新benchmark)",
            "env_name": "mix_divpen",
            "dataset_quality": "v2_b3",
            "expected_filename": "GeMS_mix_divpen_v2_b3_latent32_beta1.0_click1.0_seed58407201.ckpt"
        },
        {
            "name": "epsilon-greedy (实验组)",
            "env_name": "mix_divpen",
            "dataset_quality": "epsilon-greedy",
            "expected_filename": "GeMS_mix_divpen_epsilon-greedy_latentdim32_beta1.0_lambdaclick1.0_lambdaprior0.0_pretrained_seed58407201.ckpt"
        },
        {
            "name": "topdown epsilon-greedy",
            "env_name": "topdown_divpen",
            "dataset_quality": "epsilon-greedy",
            "expected_filename": "GeMS_topdown_divpen_epsilon-greedy_latentdim32_beta1.0_lambdaclick1.0_lambdaprior0.0_pretrained_seed58407201.ckpt"
        }
    ]

    print("=" * 80)
    print("GeMS Checkpoint 路径解析测试")
    print("=" * 80)

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {case['name']}")
        print(f"  env_name: {case['env_name']}")
        print(f"  dataset_quality: {case['dataset_quality']}")

        try:
            gems_path, lambda_click = resolve_gems_checkpoint(
                env_name=case['env_name'],
                dataset_quality=case['dataset_quality']
            )

            # 提取文件名
            actual_filename = Path(gems_path).name

            print(f"  ✅ 解析成功")
            print(f"  完整路径: {gems_path}")
            print(f"  文件名: {actual_filename}")
            print(f"  lambda_click: {lambda_click}")

            # 验证文件名是否匹配
            if actual_filename == case['expected_filename']:
                print(f"  ✅ 文件名匹配")
            else:
                print(f"  ❌ 文件名不匹配")
                print(f"     期望: {case['expected_filename']}")
                print(f"     实际: {actual_filename}")

            # 验证文件是否存在
            if Path(gems_path).exists():
                print(f"  ✅ 文件存在")
            else:
                print(f"  ❌ 文件不存在")

        except Exception as e:
            print(f"  ❌ 解析失败: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_gems_checkpoint_resolution()
