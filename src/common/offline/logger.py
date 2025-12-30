"""
轻量级 SwanLab Logger (离线RL专用)
无 PyTorch Lightning 依赖,只依赖 swanlab
"""
import swanlab
from typing import Dict, Any, Optional, List, Union
from argparse import Namespace


class SwanlabLogger:
    """
    轻量级 SwanLab Logger,专为离线RL设计

    特点:
    - 无 PyTorch Lightning 依赖
    - 接口简洁,易于使用
    - 只保留核心功能

    使用示例:
        logger = SwanlabLogger(
            project="MyProject",
            experiment_name="exp_001",
            workspace="MyWorkspace",
            config={"lr": 0.001},
            mode="cloud"
        )

        logger.log_metrics({"loss": 0.5}, step=100)
        logger.finish()
    """

    def __init__(
        self,
        project: str,
        experiment_name: str,
        workspace: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "cloud",
        logdir: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 SwanLab Logger

        Args:
            project: 项目名称
            experiment_name: 实验名称
            workspace: 工作空间名称
            description: 实验描述
            tags: 标签列表
            config: 配置字典
            mode: 运行模式 (cloud/local/offline/disabled)
            logdir: 本地日志目录
            **kwargs: 其他 swanlab.init 参数
        """
        self.project = project
        self.experiment_name = experiment_name
        self.workspace = workspace
        self.mode = mode

        # 清理配置
        clean_config = self._clean_config(config) if config else None

        # 初始化 SwanLab
        self._experiment = swanlab.init(
            project=project,
            experiment_name=experiment_name,
            workspace=workspace,
            description=description,
            config=clean_config,
            mode=mode,
            logdir=logdir,
            **kwargs
        )

    @property
    def experiment(self):
        """返回 SwanLab 实验对象"""
        return self._experiment

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ):
        """
        记录指标

        Args:
            metrics: 指标字典 {name: value}
            step: 步数 (可选)

        示例:
            logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
        """
        # 只保留数值类型的指标
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_metrics[key] = value
            elif isinstance(value, bool):
                clean_metrics[key] = int(value)

        if clean_metrics:
            if step is not None:
                self._experiment.log(clean_metrics, step=step)
            else:
                self._experiment.log(clean_metrics)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        """
        记录超参数

        Args:
            params: 超参数字典或 Namespace

        示例:
            logger.log_hyperparams({"lr": 0.001, "batch_size": 32})
        """
        if isinstance(params, Namespace):
            params = vars(params)

        clean_params = self._clean_config(params)
        if clean_params:
            self._experiment.config.update(clean_params)

    def finish(self):
        """结束实验"""
        swanlab.finish()

    @staticmethod
    def _clean_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理配置,只保留基本类型

        Args:
            config: 原始配置字典

        Returns:
            清理后的配置字典
        """
        clean = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                # 基本类型直接保留
                clean[key] = value
            elif isinstance(value, (list, tuple)):
                # 列表/元组转为字符串
                clean[key] = str(value)
            elif isinstance(value, dict):
                # 递归清理嵌套字典
                clean[key] = SwanlabLogger._clean_config(value)
            elif hasattr(value, '__dict__'):
                # 对象转为字符串
                clean[key] = str(value)
            else:
                # 其他类型转为字符串
                clean[key] = str(value)

        return clean
