import csv
import functools
import itertools
import operator
import os
import shutil
import sys
import time
import random
import typing as T
from collections import abc
from datetime import datetime
from functools import partial

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import torchzero as tz

from . import nn as mynn
from . import python_tools
from .data import DS
from .event_model.callback import Callback
from .event_model.conditional_callback import ConditionalCallback
from .learner import *
from .loaders.image import imread, imreadtensor, imwrite
from .loaders.text import txtread, txtwrite
from .loaders.csv_ import csvread, csvwrite
from .plt_tools import Fig, imshow, imshow_grid, linechart, scatter
from .python_tools import (ShutUp, clean_mem, compose, find_file_containing,
                           flatten, get0, get1, get__name__, get_all_files,
                           getlast, identity, identity_kwargs,
                           listdir_fullpaths, maybe_compose,
                           perf_counter_context, pretty_print_dict,
                           print_callable_defaults)
from .python_tools import printargs as printa
from .python_tools import reduce_dim, time_context
from .python_tools.functions import SaveSignature as sig
from .torch_tools import count_params, pad, pad_to_shape, pad_like, pad_dim
from .transforms import normalize, znormalize

CUDA = torch.device('cuda')
CPU = torch.device('cpu')