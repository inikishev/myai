#pylint:disable=redefined-outer-name
import itertools
import time
import typing as T
import warnings
from collections import abc

import torch

from ...event_model import Callback, CancelContext

if T.TYPE_CHECKING:
    from ..learner import Learner


class NoOp(Callback): pass