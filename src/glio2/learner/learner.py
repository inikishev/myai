import os
import time
import typing as T
import warnings
from collections import abc
import inspect
import numpy as np
import torch

from ..event_model import (Callback, CancelContext, ConditionalCallback,
                           EventModel)
from ..loaders.text import txtread, txtwrite
from ..loaders.yaml import yamlread, yamlwrite
from ..logger.base_logger import BaseLogger
from ..logger.dict_logger import DictLogger
from ..python_tools import (SaveSignature, get__name__, get_full_kwargs,
                            make_dict_serializeable, epoch_to_datetime)
from ..torch_tools import CUDA_IF_AVAILABLE, maybe_ensure_cpu_number
from .callbacks.default import Default

DEFAULT_CALLBACKS = ()

if T.TYPE_CHECKING:
    from accelerate import Accelerator


class Learner(EventModel):
    model: torch.nn.Module | T.Any
    loss_fn: abc.Callable
    optimizer: torch.optim.Optimizer | T.Any # type:ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler | T.Any # type:ignore

    inputs: torch.Tensor | T.Any
    """Inputs object that gets passed to the model, gets assigned before running `forward`."""
    preds: torch.Tensor | T.Any
    """Outputs of the model, gets assigned after running `forward`. Callbacks can make changes or overwrite this on `after_forward`."""
    targets: torch.Tensor | T.Any
    """Targets object that gets passed to the loss function, gets assigned before running `get_loss`."""
    loss: torch.Tensor | T.Any
    """Output of the loss function, gets assigned after running `get_loss`."""
    batch: tuple[torch.Tensor | T.Any, torch.Tensor | T.Any] | T.Any
    """Batch, gets assigned at the beginning of `one_batch`. Doesn't get assigned on `inference`."""
    dltrain: abc.Iterable
    """Train dataloader, gets assigned before starting `fit` context."""
    dl: abc.Iterable
    """Current dataloader, gets assigned before starting `one_epoch` context."""
    def __init__(
        self,
        callbacks: Callback | abc.Iterable[Callback] = (),
        model: T.Optional[torch.nn.Module | SaveSignature | abc.Callable] = None,
        loss_fn: T.Optional[abc.Callable | SaveSignature] = None,
        optimizer: T.Optional[torch.optim.Optimizer | SaveSignature | T.Any] = None, # type:ignore
        scheduler: T.Optional[torch.optim.lr_scheduler.LRScheduler | SaveSignature | T.Any] = None,

        device = CUDA_IF_AVAILABLE,
        logger: T.Optional[BaseLogger] = None,

        name: str = '{model} {loss_fn} {optimizer}{optimizer.lr} {scheduler} {postfix} - {datetime}',
        main_metric: str = 'test accuracy',

        default_callbacks: Callback | abc.Iterable[Callback] = Default(),
    ):
        super().__init__()
        self.info: dict[str, T.Any] = {"postfix": '', "info": {}}
        self.device = device
        self.accelerator: "Accelerator | T.Any" = None
        if logger is None: logger = DictLogger()
        self.logger: BaseLogger = logger
        self.name_template = name
        self.main_metric = main_metric

        self.creation_time = time.time()
        self._dirs = {}
        """Directories for each root."""

        # counters
        self.cur_epoch = 0
        """Current epoch in fit."""
        self.cur_batch = 0
        """Current batch in train or test epoch."""
        self.num_forwards = 0
        """Total number of forward passes during training."""
        self.num_backwards = 0
        """Total number of backward passes during training."""

        self.total_epochs = 0
        """Total train epochs"""
        self.total_batches = 0
        """Total train batches"""

        self.status: T.Literal['init', 'train', 'test',] = 'init'
        """Current status, gets assigned at the beginning of each epoch and batch."""

        # set all attributes
        # some of those may be SaveSignature
        # which is an easy way to save all kwargs that stuff like optimizer uses
        # all kwargs are stored in `self.info`
        for attr, cls in (('model', model), ('loss_fn', loss_fn), ('optimizer', optimizer), ('scheduler', scheduler)):
            if cls is not None: self._set_x(attr, cls)
            else: setattr(self, attr, None)

        # add all callbacks
        if isinstance(callbacks, Callback): callbacks = [callbacks]
        if isinstance(default_callbacks, Callback): default_callbacks = [default_callbacks]
        for c in default_callbacks: self.add_callback(c, default=True)
        for c in callbacks: self.add_callback(c)


    def _set_x_cls[**P](self, attr: str, x: abc.Callable[P, T.Any], *args: P.args, **kwargs: P.kwargs):
        setattr(self, attr, x(*args, **kwargs))
        self.info[attr] = {
            "name": get__name__(x),
            'params': make_dict_serializeable(get_full_kwargs(x, *args, **kwargs), raw_strings=False)
            }
        return self

    def _set_x(self, attr: str, x, params: T.Optional[abc.Mapping[str, T.Any]] = None):
        # SaveSignature contains x(*args, **kwargs) as well as signature of x
        # we set x and save signature by using _set_x_cls
        if isinstance(x, SaveSignature):
            return self._set_x_cls(attr, x.obj, **x.signature)

        # else just set the attribute
        setattr(self, attr, x)
        self.info[attr] = {"name": get__name__(x)}
        if params is not None: self.info[attr]['params'] = make_dict_serializeable(params, raw_strings=False)
        return self

    def set_model_cls[**P](self, cls: abc.Callable[P, torch.nn.Module | abc.Callable], *args: P.args, **kwargs: P.kwargs):
        return self._set_x_cls('model', cls, *args, **kwargs)
    def set_model(self, model: torch.nn.Module | abc.Callable): return self._set_x('model', model)

    def set_loss_cls[**P](self, cls: abc.Callable[P, abc.Callable], *args: P.args, **kwargs: P.kwargs):
        return self._set_x_cls('loss_fn', cls, *args, **kwargs)
    def set_loss(self, loss_fn: abc.Callable): return self._set_x('loss_fn', loss_fn)

    def set_optimizer_cls[**P](self, cls: abc.Callable[P, torch.optim.Optimizer | T.Any], *args: P.args, **kwargs: P.kwargs): # type:ignore
        return self._set_x_cls('optimizer', cls, *args, **kwargs)
    def set_optimizer(self, optimizer: torch.optim.Optimizer | T.Any): return self._set_x('optimizer', optimizer) # type:ignore

    def set_scheduler_cls[**P](self, cls: abc.Callable[P, torch.optim.lr_scheduler.LRScheduler | T.Any], *args: P.args, **kwargs: P.kwargs):
        return self._set_x_cls('scheduler', cls, *args, **kwargs)
    def set_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler | T.Any): return self._set_x('scheduler', scheduler)

    def add_named_info(self, name: str, **params: T.Any):
        """Add named misc. info, for example transforms."""
        self.info['info'][name] = make_dict_serializeable(params, raw_strings=False)

    def add_info(self, **kwargs):
        """Add misc. info"""
        self.info['info'].update(make_dict_serializeable(kwargs, raw_strings=False))
        return self

    def set_name(self, name: str = '{model} {loss_fn} {optimizer}{optimizer.lr} {scheduler} {postfix}'):
        """Sets name which is mainly used as directory name for saving stuff. Can use `{}`"""
        self.name_template = name

    def set_postfix(self, postfix: str):
        """Set postfix which may be used in the name. Can use `{}`."""
        self.info['postfix'] = postfix
        return self

    def get_main_metric(self):
        if self.main_metric in self.logger:
            return self.logger.last(self.main_metric)
        else:
            return ""

    def _process_interp_template(self, s: str) -> str:
        """Preprocesses a template inside {} brackets."""
        parts = s.rsplit('.', 1)
        base = parts[0]
        if len(parts) == 2: attr = parts[1]
        else: attr = None

        # get from self.info
        if base in self.info:
            if 'name' in self.info[base]:
                if attr is None: return self.info[base]['name']
                else:
                    if 'params' in self.info[base] and attr in self.info[base]['params']: return str(self.info[base]['params'][attr])
                    elif attr in self.info[base]: return str(self.info[base][attr])
                    else: return ''

            elif attr is None: return str(self.info[base])
            else:
                if attr in self.info[base]: return str(self.info[base][attr])
                else: return ''

        # get from logger
        elif base == 'logger':
            if attr is None: raise ValueError(f'Invalid template: {s}, {attr} not found in logger')
            elif attr in self.logger: return self.logger.last(attr)
            else: return ''

        # get some other attribute
        else:
            if base == 'datetime': return epoch_to_datetime(self.creation_time).strftime("%Y.%m.%d %H-%M-%S")
            elif base == 'main_metric':
                v = maybe_ensure_cpu_number(self.get_main_metric())
                if isinstance(v, float): v = f'{v:.4f}'
                return str(v)
            elif base in dir(self):
                v = getattr(self, base)
                if v is not None: return str(v)
                return ''
            else: raise ValueError(f'Invalid template: {s}, {base} not found')

    def process_template(self, template: str) -> str:
        """Processes a template like `'{total_epochs} {logger.test loss} {main_metric}'`"""
        template = template.replace('{{', '__BRACEOPEN__').replace('}}', '__BRACECLOSE__')
        # "stuff {attr1} {attr2}"
        starts = template.split('{')
        for i, s in enumerate(starts.copy()):
            if '}' in s:
                interp = s[:s.find('}')]
                starts[i] = starts[i].replace(f'{interp}}}', self._process_interp_template(interp))

        string = ''.join(starts).replace('__BRACEOPEN__', '{').replace('__BRACECLOSE__', '}')
        while '  ' in string: string = string.replace('  ', ' ')
        return string.strip()

    def get_learner_dir(self, root: str = 'runs'):
        """Creates a directory if it doesn't exist and returns the path. The path is `root/name`"""
        if root in self._dirs: return self._dirs[root]

        if not os.path.exists(root): os.mkdir(root)
        dir = os.path.join(root, self.name)
        if os.path.exists(dir):
            dir = f'{dir} 2'
            c = 2
            while os.path.exists(dir):
                c += 1
                dir = f'{dir[:-2]} {c}'

        os.mkdir(dir)
        self._dirs[root] = dir
        return dir


    def get_epoch_dir(self, root: str = 'runs', epoch_template = '{total_epochs} {logger.test loss} {main_metric}'):
        """Creates a directory if it doesn't exist and returns the path. The path is `root/name/epoch`"""
        root = self.get_learner_dir(root)
        dir = os.path.join(root, self.process_template(epoch_template))
        if not os.path.exists(dir): os.mkdir(dir)
        return dir

    def get_prefix_epoch_dir(self, prefix: str, root: str = 'runs', epoch_template = '{total_epochs} {logger.test loss} {main_metric}'):
        """Creates a directory if it doesn't exist and returns the path. The path is `root/name/prefix/epoch`"""
        root = self.get_learner_dir(root)
        rootprefix = os.path.join(root, prefix)
        if not os.path.exists(rootprefix): os.mkdir(rootprefix)
        dir = os.path.join(rootprefix, self.process_template(epoch_template))
        if not os.path.exists(dir): os.mkdir(dir)
        return dir


    @property
    def name(self):
        """By default this is `{model} {loss_fn} {optimizer}{optimizer.lr} {scheduler} {postfix} {datetime}`"""
        n = self.process_template(self.name_template)
        if len(n) == 0 or n == ' ': n = 'empty-name'
        return n

    def set_use_closure(self, use_closure: bool):
        """Whether to pass closure to optimizer. When Learner is created, this is set to True by default."""
        cb: Default = self.get_callback('DefaultCB') # type:ignore
        cb._use_closure = use_closure

    def state_dict(self):
        """State dict. Saves the following attributes: ones that have `state_dict`,
        and ones that are `int, float, str, bool, None, np.ndarray, torch.Tensor`.
        This includes attributes like total_batches, etc."""
        state_dict = {}
        for attr in dir(self):

            # skip private methods
            if attr.startswith('_') or attr == 'callbacks': continue

            # skip properties
            if attr in dir(type(self)) and isinstance(getattr(type(self), attr), property): continue

            # get the atttribute
            x = getattr(self, attr)

            # skip methods
            if inspect.ismethod(x): continue

            # if it has a state_dict, save a dictionary with the state dict and the type
            if hasattr(x, 'state_dict'):
                state_dict[attr] = {"state_dict": x.state_dict(), "type": get__name__(x)}

            # if it is a serializeable object,
            elif isinstance(x, (int, float, str, bool, np.ndarray, torch.Tensor)) or x is None:
                state_dict[attr] = {"object": x}

            # else we store names and types of those objects so that they can be restored
            else:
                state_dict[attr] = {"type": get__name__(x)}
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Load a state dict."""
        for attr, value in state_dict.items():
            # load state_dict
            if 'state_dict' in value:
                self_attr = getattr(self, attr)
                if get__name__(self_attr) != value['type']:
                    warnings.warn(
                        f"Loading state dict for {attr} but type is different. \
                        Self.{attr} is {get__name__(self_attr)}, state_dict['{attr}'] is {value['type']}"
                    )
                getattr(self, attr).load_state_dict(value['state_dict'])

            # or just set object to its value
            elif 'object' in value:
                setattr(self, attr, value['object'])


    def save(self, dir: str, mkdir = True):
        """Saves this learner to a directory, creates files in that directory."""
        if not os.path.exists(dir) and mkdir: os.mkdir(dir)

        # save info
        info = self.info.copy()
        info['cur_batch'] = self.cur_batch; info['cur_epoch'] = self.cur_epoch
        info['total_batches'] = self.total_batches; info['total_epochs'] = self.total_epochs
        info['num_forwards'] = self.num_forwards; info['num_backwards'] = self.num_backwards
        yamlwrite(info, os.path.join(dir, 'info.yaml'))

        # save state_dicts
        learner_attrs = {} # learner state_dict for stuff like cur_batch
        for k, v in self.state_dict().items():
            if k == 'logger': continue # logger is saved using `save`
            elif 'state_dict' in v: torch.save(v['state_dict'], os.path.join(dir, f'{k}.state_dict'))
            else: learner_attrs[k] = v
        torch.save(learner_attrs, os.path.join(dir, 'learner.attrs'))

        # save logger
        self.logger.save(os.path.join(dir, 'logger.npz'))

        # save model and optimizer as strings
        txtwrite(str(self.model), os.path.join(dir, 'model.txt'))
        txtwrite(str(self.optimizer), os.path.join(dir, 'optimizer.txt'))

    def load(self, dir: str):
        files = set(os.listdir(dir))
        if 'info.yaml' in files: self.info = yamlread(os.path.join(dir, 'info.yaml'))
        if 'logger.npz' in files: self.logger.load(os.path.join(dir, 'logger.npz'))

        # load attrs like cur_batch
        if 'learner.attrs' in files:
            learner_attrs: dict[str, T.Any] = torch.load(os.path.join(dir, 'learner.attrs'), weights_only = False)
            for k, v in learner_attrs.items():
                if 'object' in v:
                    setattr(self, k, v['object'])

        # load state_dicts
        for file in files:
            if file.endswith('.state_dict'):
                attr_name = file.replace('.state_dict', '')
                attr = getattr(self, attr_name)
                try: attr.load_state_dict(torch.load(os.path.join(dir, file), weights_only = False), )
                except Exception as e: warnings.warn(f"Failed to load state dict for {attr_name}: {e!r}")

    @property
    def training(self):
        return self.model.training
    # ---------------------------------------------------------------------------- #
    #                            callback based methods                            #
    # ---------------------------------------------------------------------------- #

    def log(self, metric: str, value: T.Any):
        self.fire_event('log', metric, value)

    def train(self):
        self.fire_event('train',)

    def eval(self):
        self.fire_event('eval',)

    def forward(self, inputs,):
        """Pass inputs through model and return predictions."""
        self.inputs = inputs
        self.preds = self.fire_event('forward', self.inputs)
        if self.status == 'train': self.num_forwards += 1
        self.fire_event('after_forward')
        return self.preds

    def get_loss(self, *args):
        """Evaluate loss value between preds and targets."""
        if len(args) == 2: self.targets = args[1]
        self.loss = self.fire_event('get_loss', *args)
        return self.loss

    def backward(self, loss: torch.Tensor, **kwargs):
        """Call backward on loss"""
        self.fire_event('backward', loss, **kwargs)
        self.fire_event('after_backward')
        if self.status == 'train': self.num_backwards += 1

    def zero_grad(self, set_to_none: bool = True):
        """Zero grad"""
        self.fire_event('zero_grad', set_to_none = set_to_none)

    def closure(self, batch, backward=True) -> torch.Tensor:
        self.fire_event('before_any_step')
        self.fire_event(f'before_{self.status}_step')
        loss = self.fire_event('closure', batch = batch, backward=backward)
        self.fire_event('after_any_step')
        self.fire_event(f'after_{self.status}_step')
        return loss

    def make_closure(self, batch) -> abc.Callable[[], torch.Tensor]:
        return self.fire_event('make_closure', batch = batch)

    def inference(self, inputs, enable_grad = False):
        return self.fire_event('inference', inputs, enable_grad)

    def optimizer_step(self, *args, **kwargs):
        self.fire_event('optimizer_step', *args, **kwargs)
        self.fire_event('after_optimizer_step')

    def one_batch(self, batch, train: bool):
        self.batch = batch

        self.status = 'train' if train else 'test'

        self.fire_event('before_any_batch')
        self.fire_event(f'before_{self.status}_batch')

        self.fire_event('one_batch', batch = self.batch, train = train)
        if train: self.total_batches += 1

        self.fire_event(f'after_{self.status}_batch')
        self.fire_event('after_any_batch')

    def one_epoch(self, dl: abc.Iterable, train: bool):
        self.dl = dl
        self.status = 'train' if train else 'test'

        self.fire_event('before_any_epoch')
        self.fire_event(f'before_{self.status}_epoch')

        # run epoch context which catches CancelContext("epoch")
        with self.context('epoch', after = ('after_any_epoch', f'after_{self.status}_epoch')):
            self.fire_event('one_epoch', dl = self.dl, train = train)
            if train: self.total_epochs += 1
        # context runs `after_x` events on exit there

    def fit(self, dltrain: abc.Iterable, n_epochs: int, catch_kb_interrupt = True, ):
        # attrs
        self.dltrain = dltrain
        self.n_epochs = n_epochs
        self.epochs_iterator: abc.Iterable[int] = range(n_epochs)
        self.catch = [KeyboardInterrupt, ] if catch_kb_interrupt else []

        self.fire_event('before_fit')

        # run fit context which catches CancelContext("fit")
        try:
            with self.context('fit', after = ('after_fit', ), catch = tuple(self.catch)):
            # we pass dltrain and epochs_iterator to the event because
            # they can be replaced by other callbacks that are used
            # in the middle of training, and we don't want fit callback to break
                self.fire_event('fit', dltrain = self.dltrain, epochs_iterator = self.epochs_iterator, )
        except tuple(self.catch): pass
        except Exception as e:
            self.fire_event('on_fit_exception')
            raise e

        # context runs `after_fit` event on exit, CancelContext('fit') and optionally KeyboardInterrupt