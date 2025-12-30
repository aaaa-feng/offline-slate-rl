"""
GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
"""

from __future__ import annotations

import math
from argparse import Namespace
from io import BytesIO
from numbers import Number
from typing import Any, Dict, Optional, Sequence, Union

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False

try:
    from plotly.basedatatypes import BaseFigure as PlotlyBaseFigure  # type: ignore
    import plotly.io as plotly_io  # type: ignore
except Exception:  # pragma: no cover - plotly is optional
    PlotlyBaseFigure = None
    plotly_io = None

from PIL import Image as PILImage  # type: ignore

import swanlab
from swanlab import Image as SwanImage
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class SwanlabLogger(LightningLoggerBase):
    """
    Minimal PyTorch Lightning logger that writes metrics and media to SwanLab.
    """

    LOGGER_JOIN_CHAR = "/"

    def __init__(
        self,
        project: Optional[str] = None,
        experiment_name: Optional[str] = None,
        workspace: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        logdir: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: Optional[Union[str, bool]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.project = project or "GeMS"
        self.experiment_name = experiment_name
        self.workspace = workspace
        self.description = description
        self.tags = list(tags) if tags else None
        self.mode = mode
        self.logdir = logdir
        self.run_id = run_id
        self.resume = resume
        self._initial_config = config or {}
        self._prefix = prefix

        self._experiment: Optional[swanlab.Run] = None

    @property
    @rank_zero_experiment
    def experiment(self) -> swanlab.Run:
        if self._experiment is None:
            init_kwargs: Dict[str, Any] = {"project": self.project}
            if self.workspace:
                init_kwargs["workspace"] = self.workspace
            if self.experiment_name:
                init_kwargs["experiment_name"] = self.experiment_name
            if self.description:
                init_kwargs["description"] = self.description
            if self.tags:
                init_kwargs["tags"] = list(self.tags)
            if self.mode:
                init_kwargs["mode"] = self.mode
            if self.logdir:
                init_kwargs["logdir"] = self.logdir
            if self.run_id:
                init_kwargs["id"] = self.run_id
            if self.resume is not None:
                init_kwargs["resume"] = self.resume
            if self._initial_config:
                init_kwargs["config"] = self._initial_config
            self._experiment = swanlab.init(**init_kwargs)
        return self._experiment

    @property
    def name(self) -> str:
        return self.project

    @property
    def version(self) -> Union[int, str, None]:
        if self._experiment is not None:
            return getattr(self._experiment, "id", None)
        return self.run_id

    @property
    def save_dir(self) -> Optional[str]:
        return self.logdir

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        params = self._sanitize_params(params)
        if not params:
            return
        self.experiment.config.update(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        metrics = self._add_prefix(metrics)
        payload: Dict[str, float] = {}
        for key, value in metrics.items():
            scalar = self._to_float(value)
            if scalar is not None and not (isinstance(scalar, float) and math.isnan(scalar)):
                payload[key] = scalar
        if not payload:
            return
        if step is not None:
            self.experiment.log(payload, step=int(step))
        else:
            self.experiment.log(payload)

    @rank_zero_only
    def log_figure(self, key: str, figure: Any, step: Optional[int] = None, caption: Optional[str] = None) -> None:
        image = self._to_swan_image(figure, caption)
        if image is None:
            return
        payload_key = f"{self._prefix}{self.LOGGER_JOIN_CHAR}{key}" if self._prefix else key
        data = {payload_key: image}
        if step is not None:
            self.experiment.log(data, step=int(step))
        else:
            self.experiment.log(data)

    def finalize(self, status: str) -> None:  # type: ignore[override]
        super().finalize(status)
        if self._experiment is not None:
            try:
                swanlab.finish()
            finally:
                self._experiment = None

    # -----------------------
    # Internal helper methods
    # -----------------------
    def _to_float(self, value: Any) -> Optional[float]:
        if isinstance(value, Number):
            return float(value)
        if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().cpu().mean().item())
        return None

    def _to_swan_image(self, figure: Any, caption: Optional[str]) -> Optional[SwanImage]:
        if isinstance(figure, SwanImage):
            return figure
        if PlotlyBaseFigure is not None and isinstance(figure, PlotlyBaseFigure) and plotly_io is not None:
            try:
                raw = plotly_io.to_image(figure, format="png")
                with BytesIO(raw) as buffer:
                    pil_image = PILImage.open(buffer).convert("RGB")
                    return SwanImage(pil_image, caption=caption)
            except Exception:
                return None
        try:
            return SwanImage(figure, caption=caption)
        except Exception:
            return None

