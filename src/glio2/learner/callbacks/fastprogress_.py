import itertools
import time
import typing as T
import warnings
from collections import abc

import torch
from fastprogress import master_bar, progress_bar

from ...event_model import Callback

if T.TYPE_CHECKING:
    from ..learner import Learner


class FastProgress(Callback):
    order = 10

    def __init__(
        self,
        metrics: str | abc.Iterable[str] = ("train loss", "test loss"),
        bar_sec: float = 1,
        plot_sec: float = 10,
        ybounds: T.Optional[tuple[float | None, float | None] | abc.Sequence[float | None]] = None,
    ):
        self.metrics = metrics
        self.bar_step = bar_sec
        self.plot_step = plot_sec
        self.last_bar_update = time.time()
        self.last_plot_update = self.last_bar_update
        self.ybounds = ybounds
        self.initialized = False

    def before_fit(self, learner: "Learner"):
        learner.epochs_iterator = self.mb = master_bar(learner.epochs_iterator)
        self.pb = progress_bar(itertools.count(), total = 1, parent=self.mb) # type:ignore

    def _plot(self, learner: "Learner"):
        graphs = []
        for name in self.metrics:
            if name in learner.logger:
                graphs.append([list(learner.logger[name].keys()), list(learner.logger[name].values())])
            elif learner.total_epochs >= 1: # to avoid triggering this on 1st epoch when metrics havent been created
                warnings.warn(f"metric {name} not found in logger")

        self.mb.update_graph(graphs, None, self.ybounds)

    def after_any_step(self, learner: "Learner"):
        if not self.initialized:
            self.pb.update(0) # required
            self.initialized = True

        t = time.time()
        # update the bar every second
        if t - self.last_bar_update > self.bar_step:
            self.pb.total = len(learner.dl) # type:ignore
            self.pb.comment = f'{learner.status} s{learner.num_forwards}; {", ".join(f"{i} = {learner.logger.last(i):.3f}" for i in self.metrics if i in learner.logger)}'
            self.pb.update(learner.cur_batch)
            self.last_bar_update = t
        # update the plot every 10 seconds
        if t - self.last_plot_update > self.plot_step:
            self._plot(learner)
            self.last_plot_update = t

    def after_fit(self, learner: "Learner"):
        self._plot(learner)
