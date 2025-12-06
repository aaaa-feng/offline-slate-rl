"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
"""

from argparse import ArgumentParser

class MyParser(ArgumentParser):
    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


class MainParser(MyParser):
    def __init__(self):
        # 1. 修改这里：使用 super() 正确初始化，这样才能用 str2bool
        super().__init__()
        
        #   ---- General parameters ----   #
        self.add_argument(
            "--exp_name", type=str, default="test_exp", help="Experiment name."
        )
        self.add_argument(
            "--run_name", type=str, default="test_run", help="Run name."
        )
        self.add_argument(
            "--data_dir", type=str, default="data/", help="Path to data/results parent directory."
        )
        self.add_argument(
            "--device", type=str, default="cpu", help="PyTorch device."
        )
        self.add_argument(
            "--seed", type=int, default=2021, help="Seed for reproducibility."
        )
        self.add_argument(
            "--verbose", type=self.str2bool, default=False, help="Print for debugging."
        )
        self.add_argument(
            "--max_steps", type=int, default=1e6, help="Max number of agent training steps to perform"
        )
        self.add_argument(
            "--iter_length_agent", type=int, default=-1, help="Number of agent training episodes per iteration."
        )
        self.add_argument(
            "--iter_length_ranker", type=int, default=1000, help="Number of ranker training steps to perform per iteration."
        )
        self.add_argument(
            "--n_iter", type=int, default=100, help="Number of training iterations to perform for rankers which require it."
        )
        self.add_argument(
            "--val_check_interval", type=int, default=300, help="Number of training steps to perform between each validation epoch.(Unused)"
        )
        self.add_argument(
            "--check_val_every_n_epoch", type=int, default=25, help="Number of training epochs to perform between each validation epoch."
        )
        self.add_argument(
            "--name", type=str, default="default", help="Used to easily get legend on plots."
        )

        #   ---- Buffer parameters ----   #
        self.add_argument(
            "--capacity", type=int, default=1000000, help="Capacity of the buffer."
        )
        self.add_argument(
            "--batch_size", type=int, default=32, help="Minibatch size for RL update."
        )

        #   ---- Environment parameters ----   #
        self.add_argument(
            "--env_name", type=str, default="CartPole-v0", help="Gym environment ID."
        )

        #   ---- Ranker parameters ----   #
        # 【重要】：这里删除了 --ranker_dataset
        # 原因：rankers.py 中的 GeMS 类会自动添加这个参数。
        # 如果这里保留，就会报 "conflicting option string" 错误。

        #   ---- Logging and printing parameters ----   #
        self.add_argument(
            "--log_every_n_steps", type=int, default=1, help="Frequency of metric logging."
        )
        self.add_argument(
            "--progress_bar", type=self.str2bool, default=True, help="Toggle progress bar."
        )
        self.add_argument(
            "--swan_project", type=str, default=None, help="Override SwanLab project name (defaults to exp_name)."
        )
        self.add_argument(
            "--swan_workspace", type=str, default=None, help="Optional SwanLab workspace."
        )
        self.add_argument(
            "--swan_mode",
            type=str,
            default=None,
            choices=["cloud", "local", "offline", "disabled"],
            help="SwanLab logging mode override.",
        )
        self.add_argument(
            "--swan_tags",
            type=str,
            nargs="*",
            default=None,
            help="Additional SwanLab tags (space separated).",
        )
        self.add_argument(
            "--swan_description", type=str, default=None, help="Description for SwanLab experiments."
        )
        self.add_argument(
            "--swan_logdir", type=str, default=None, help="Local directory for SwanLab run artifacts."
        )
        self.add_argument(
            "--swan_run_id",
            type=str,
            default=None,
            help="Explicit SwanLab run id (enables resuming specific runs).",
        )
        self.add_argument(
            "--swan_resume",
            type=str,
            default=None,
            choices=["must", "allow", "never"],
            help="Resume policy when reusing a SwanLab run id.",
        )
        #   ---- Checkpoint Strategy parameters (New) ----   #
        self.add_argument(
            "--save_every_n_steps", type=int, default=0,
            help="Save a checkpoint every N steps. If 0, only the best model (by val_reward) is saved."
        )
        self.add_argument(
            "--save_path", type=str, default=None,
            help="Custom directory to save checkpoints (overrides default)."
        )

        #   ---- Experiment Organization parameters (New) ----   #
        self.add_argument(
            "--exp_purpose", type=str, default="experiment",
            help="Purpose/description of this experiment (e.g., 'medium_collection', 'baseline_test'). Used in log directory naming."
        )