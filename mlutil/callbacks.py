import abc
from dataclasses import dataclass, field
import livelossplot
from typing import List, Dict
import json
import neptune.new as neptune


class LoggingCallback(abc.ABC):
    def __call__(self, metrics: Dict[str, float]):
        """log metrics"""


class MetricListCallback(LoggingCallback):
    def __init__(self):
        self.log = []

    def __call__(self, metrics):
        self.log.append(metrics)


class LivelossplotCallback(LoggingCallback):
    def __init__(self, n_logging_steps: int):
        self.n_logging_steps = n_logging_steps
        self.liveplot = livelossplot.PlotLosses()
        self.i = 0

    def __call__(self, metrics):
        self.liveplot.update(metrics)
        if self.i % self.n_logging_steps == 0:
            self.liveplot.send()
        self.i += 1


class NeptuneCallback(LoggingCallback):
    def __init__(
        self,
        tags: List[str],
        neptune_config_path: str = None,
        api_token: str = None,
        project: str = None,
    ):
        self.run = self.init_neptune(tags, neptune_config_path, api_token, project)

    def __call__(self, metrics):
        for param_name in metrics.keys():
            self.run[param_name].log(metrics[param_name])

    def init_neptune(
        tags: List[str] = [],
        neptune_config_path: str = None,
        api_token: str = None,
        project: str = None,
    ):
        configured_by_vars = api_token is not None and project is not None
        assert neptune_config_path is not None or configured_by_vars
        if neptune_config_path is not None:
            neptune_args = json.loads(open(neptune_config_path, "r").read())
            return neptune.init_run(tags=tags, **neptune_args)
        else:
            return neptune.init_run(tags=tags, api_token=api_token, project=project)


class MultiCallback(LoggingCallback):
    def __init__(self, callbacks: List[LoggingCallback]):
        self.callbacks = callbacks

    def __call__(self, metrics):
        for cb in self.callbacks:
            cb(metrics)
