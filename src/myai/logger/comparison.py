import logging
import operator
import os
import typing as T
import warnings
from abc import ABC, abstractmethod
from collections import abc
from collections.abc import Iterator, Mapping, MutableMapping

import numpy as np
import torch
from glio.python_tools import int_at_beginning_raise

from ..plt_tools._types import _K_Line2D
from ..plt_tools.fig import Fig
from .base_logger import BaseLogger
from .dict_logger import DictLogger


class Comparison:
    def __init__(self, loggers: abc.Mapping[str, BaseLogger]):
        self.loggers = dict(loggers)

    def add_runs_dir(self, dir: str):
        loggers = {}
        for run_name in os.listdir(dir):
            run = os.path.join(dir, run_name) # get full path
            # check if run has checkpoints
            # note that checkpoints may be in a epochs folder as well
            # but I can't be bothered to implement that now
            if 'checkpoint' in os.listdir(run):
                checkpoint = os.path.join(run, 'checkpoint')
                # then check if the checkpoint has a logger saved in it
                if 'logger.npz' in os.listdir(checkpoint):
                    loggers[run_name] = DictLogger.from_file(os.path.join(checkpoint, 'logger.npz'))

            # checkpoints are inside epoch folders
            else:
                dirs = [i for i in os.listdir(run) if i[0].isnumeric()]
                dirs.sort(key = int_at_beginning_raise)
                for d in dirs[::-1]:
                    if 'checkpoint' in os.listdir(os.path.join(run, d)):
                        checkpoint = os.path.join(run, d, 'checkpoint')
                        # then check if the checkpoint has a logger saved in it
                        if 'logger.npz' in os.listdir(checkpoint):
                            loggers[run_name] = DictLogger.from_file(os.path.join(checkpoint, 'logger.npz'))
                            break


        self.loggers.update(loggers)

    @classmethod
    def from_runs_dir(cls, dir:str):
        comparison = cls({})
        comparison.add_runs_dir(dir)
        if len(comparison.loggers) == 0:
            raise FileNotFoundError(f'No loggers found in {dir}')
        return comparison

    def n_highest(self, metric, n: int, last = False):
        if last: caller = operator.methodcaller('last', metric)
        else: caller = operator.methodcaller('max', metric)

        max_values = sorted([(k, caller(v)) for k,v in self.loggers.items()], key = lambda x:x[1], reverse=True)
        return Comparison({k:self.loggers[k] for k,_ in max_values[:n]})

    def n_lowest(self, metric, n: int, last = False):
        if last: caller = operator.methodcaller('last', metric)
        else: caller = operator.methodcaller('min', metric)

        min_values = sorted([(k, caller(v)) for k,v in self.loggers.items()], key = lambda x:x[1])
        return Comparison({k:self.loggers[k] for k,_ in min_values[:n]})

    def n_best(self, metric, n: int, highest = True, last = False):
        if highest: return self.n_highest(metric, n, last)
        return self.n_lowest(metric, n, last)

    def plot(self, metric: str, n: int | None = 13, highest = True, last = False, x = None, fig = None, **kwargs: T.Unpack[_K_Line2D]):
        if n is None: comp = self
        else: comp = self.n_best(metric, n, highest, last)

        k: dict[str, T.Any] = kwargs.copy() # type:ignore # this is necesary for pylance to shut up
        if fig is None: fig = Fig()
        xlabel, ylabel = None, None
        for name, logger in comp.loggers.items():
            if metric not in logger:
                logging.warning('%s is not in %s', metric, name)
            else:
                if x is None:
                    xvals = list(logger[metric].keys())
                    yvals = list(logger[metric].values())
                    xlabel, ylabel = 'step', metric
                else:
                    xlabel, ylabel = x, metric
                    xvals, yvals = logger.get_shared_metrics(x, metric)

                fig.linechart(xvals, yvals, label = name, **k)
        return fig.axlabels(xlabel, ylabel).legend().ticks().grid()

    def linechart(self, x: str, y: str, n: int | None = None, highest = True, last = False, fig = None, **kwargs: T.Unpack[_K_Line2D]):
        if n is None: comp = self
        else: comp = self.n_best(y, n, highest, last)

        k: dict[str, T.Any] = kwargs.copy() # type:ignore # this is necesary for pylance to shut up
        if fig is None: fig = Fig()
        for name, logger in comp.loggers.items():
            if x not in logger or y not in logger:
                logging.warning('%s or %s is not in %s', x, y, name)
            else:
                xvals = logger.get_metric_interpolate(x)
                yvals = logger.get_metric_interpolate(y)

                fig.linechart(xvals, yvals, label = name, **k)
        return fig.axlabels(x, y).legend().ticks().grid()

