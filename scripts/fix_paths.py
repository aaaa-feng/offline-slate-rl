#!/usr/bin/env python3
"""
自动修复硬编码路径的脚本
将所有硬编码路径替换为使用 config.paths 模块
"""
import os
import re
from pathlib import Path

# 需要替换的硬编码路径模式
HARDCODED_PATTERNS = [
    (r'/data/liyuefeng/gems/gems_official/official_code', 'PROJECT_ROOT'),
    (r'data/RecSim/embeddings', 'EMBEDDINGS_DIR'),
    (r'data/checkpoints', 'CHECKPOINTS_DIR'),
    (r'offline_datasets', 'OFFLINE_DATASETS_DIR'),
    (r'logs/', 'LOGS_DIR'),
    (r'swanlog/', 'SWANLOG_DIR'),
]

def fix_file_paths(file_path):
    """修复单个文件中的硬编码路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        modified = False

        # 检查是否需要添加 import
        needs_import = False

        # 替换硬编码路径
        for pattern, replacement in HARDCODED_PATTERNS:
            if pattern in content:
                needs_import = True
                # 根据不同的模式进行替换
                if pattern == '/data/liyuefeng/gems/gems_official/official_code':
                    # 替换绝对路径
                    content = content.replace(f'"{pattern}"', f'str(paths.PROJECT_ROOT)')
                    content = content.replace(f"'{pattern}'", f'str(paths.PROJECT_ROOT)')
                else:
                    # 替换相对路径
                    content = content.replace(f'"{pattern}"', f'str(paths.{replacement})')
                    content = content.replace(f"'{pattern}'", f'str(paths.{replacement})')
                modified = True

        # 如果需要导入且还没有导入
        if needs_import and 'from config import paths' not in content and 'import paths' not in content:
            # 在文件开头添加 import（在其他 import 之后）
            lines = content.split('\n')
            import_index = 0

            # 找到最后一个 import 语句的位置
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_index = i + 1

            # 插入新的 import
            lines.insert(import_index, 'from config import paths')
            content = '\n'.join(lines)
            modified = True

        # 如果内容被修改，写回文件
        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"❌ 处理文件失败 {file_path}: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("自动修复硬编码路径")
    print("=" * 80)

    # 获取 code/src 目录
    code_root = Path(__file__).resolve().parent.parent
    src_dir = code_root / "src"

    print(f"\n扫描目录: {src_dir}")

    # 查找所有 Python 文件
    python_files = list(src_dir.rglob("*.py"))

    print(f"找到 {len(python_files)} 个 Python 文件\n")

    modified_count = 0

    for file_path in python_files:
        if fix_file_paths(file_path):
            print(f"✅ 修复: {file_path.relative_to(code_root)}")
            modified_count += 1

    print(f"\n" + "=" * 80)
    print(f"完成！共修复 {modified_count} 个文件")
    print("=" * 80)

if __name__ == "__main__":
    main()
