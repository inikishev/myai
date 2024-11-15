import typing as T
from collections.abc import Iterable
import torch

from ...event_model import ConditionalCallback

if T.TYPE_CHECKING:
    from ..learner import Learner


class Checkpoint(ConditionalCallback):
    order = 1000
    def __init__(self, state_dict=True, logger = True, info = True, text = True, root = 'runs', in_epoch_dir = True):
        super().__init__()
        self.state_dict = state_dict
        self.logger = logger
        self.info = info
        self.text = text
        self.root = root
        self.in_epoch_dir = in_epoch_dir

    def __call__(self, learner: 'Learner'):
        if self.in_epoch_dir: dir = learner.get_epoch_dir(self.root, postfix='checkpoint')
        else: dir = learner.get_learner_dir(self.root, postfix='checkpoint')

        learner.save(dir, mkdir=False, state_dict=self.state_dict, logger=self.logger, info=self.info, text = self.text)