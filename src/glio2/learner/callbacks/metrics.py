import time
import typing as T
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import torch

from ...event_model import Callback
from ...torch_tools.conversion import maybe_ensure_detach_cpu
from ...metrics import accuracy

if T.TYPE_CHECKING:
    from ..learner import Learner


class LogLoss(Callback):
    order = -1
    def __init__(self, agg_fn = np.mean, name = 'loss'):
        self.test_losses = []
        self.agg_fn = agg_fn
        self.name = name

    def after_train_step(self, learner: "Learner"):
        learner.log(f"train {self.name}", learner.loss.detach().cpu())

    def after_test_step(self, learner: "Learner"):
        self.test_losses.append(learner.loss.detach().cpu())

    def after_test_epoch(self, learner: "Learner"):
        if len(self.test_losses) > 0:
            learner.log(f"test {self.name}", np.mean(self.test_losses))
            self.test_losses = []


class LogTime(Callback):
    order = 1000
    def after_train_batch(self, learner: "Learner"):
        learner.log("time", time.time())


class Metric(Callback, ABC):
    order = -1
    """A metric that gets logged after train and test batches.
    Please make sure order is not bigger than 0, reason being that
    some callbacks might run stuff through the model and assign `preds` and `targets`.
    So the metric won't run with the actual train or test samples."""
    def __init__(self, metric: str, train_step: int | None, test_step: int | None, agg_fn = np.mean):
        super().__init__()
        self.metric = metric
        self.train_step = train_step
        self.test_step = test_step
        self.agg_fn = agg_fn

        self.test_epoch_values = []
        if self.order > 0:
            warnings.warn(
                f"Metric {self.metric} has order {self.order} which is higher than 0, so it might run after callbacks that modify `learner.preds`, `learner.targets`, etc."
            )

    @abstractmethod
    def __call__(self, learner: "Learner") -> T.Any:
        """Evaluate the metric. Please make sure returned value is detached and on CPU."""

    def after_train_step(self, learner: "Learner"):
        if self.train_step is not None and learner.num_forwards % self.train_step == 0:
            learner.log(f'train {self.metric}', self(learner))

    def after_test_step(self, learner: "Learner"):
        if self.test_step is not None and learner.cur_batch % self.test_step == 0:
            self.test_epoch_values.append(self(learner))

    def after_test_epoch(self, learner: "Learner"):
        if len(self.test_epoch_values) > 0:
            learner.log(f'test {self.metric}', self.agg_fn(self.test_epoch_values))
            self.test_epoch_values = []

class Accuracy(Metric):
    def __init__(self, train_step: int | None = 1, test_step: int | None = 1, agg_fn = np.mean, name = 'accuracy'):
        super().__init__(name, train_step, test_step, agg_fn)

    def __call__(self, learner: "Learner") -> float:
        return accuracy(learner.preds, learner.targets).detach().cpu().item()