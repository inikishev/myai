import os
import typing as T
from collections import abc

import numpy as np
import torch
from torchvision.utils import make_grid

from ...event_model import Callback, ConditionalCallback
from ...transforms import normalize as _normalize, force_hw3
from ...torch_tools import pad_to_shape, overlay_segmentation, make_segmentation_overlay
from ...video import OpenCVRenderer

if T.TYPE_CHECKING:
    from ..learner import Learner

class Renderer(Callback):
    order = 11

    def __init__(
        self,
        root="runs",
        name = 'outputs.mp4',
        nrows=2,
        fps = 60,
        codec = 'mp4v',
        id = 0,
    ):
        super().__init__()
        self.root = root
        self.name = name
        self.nrows = nrows
        self.fps = fps
        self.codec = codec
        self.id = id

        self.renderer = None
        self.tiles = []

    def add_video_tile(self, learner: "Learner", tile: torch.Tensor | None, id = 0):
        if id == self.id:
            if tile is not None:
                tile = tile.squeeze()
                if tile.ndim not in (2, 3): raise ValueError(f"tile must be 2 or 3 dimensional, got {tile.shape}")
            self.tiles.append(tile)

    def after_forward(self, learner: "Learner"):
        if learner.status == 'train':
            if self.renderer is None:
                self.renderer = OpenCVRenderer(
                    os.path.join(learner.get_learner_dir(self.root), "outputs.mp4"),
                    fps=self.fps,
                    codec=self.codec,
                )

            # compose the final frame
            total = [i for i in self.tiles if i is not None]
            if len(total) > 1:

                # pad all images to the size of the largest image
                max_x = max([i.shape[-1] for i in total])
                max_y = max([i.shape[-2] for i in total])
                # all images must have the same shape so we make them 3hw
                total = [force_hw3(
                    pad_to_shape(i, shape = (max_y, max_x), mode='min', where='center')
                ).moveaxis(-1, 0) for i in total]
                frame = make_grid(total, nrow=self.nrows).detach().cpu()

            else: frame = total[0]

            # add the normalized frame
            self.renderer.add_frame(frame.clip(0,255).detach().cpu().numpy().astype(np.uint8, copy=False))

            self.tiles = []

    def exit(self, learner: "Learner"):
        if self.renderer is not None: self.renderer.release()
        self.renderer = None

    def after_fit(self, learner: "Learner"):
        if self.renderer is not None: self.renderer.release()
        self.renderer = None

    def on_fit_exception(self, learner: "Learner"):
        if self.renderer is not None: self.renderer.release()
        self.renderer = None


class _BatchCatVideoCallback(Callback):
    order = 10
    def __init__(
        self,
        inputs: torch.Tensor | None,
        targets: torch.Tensor | None,
        n: int | None,
        nrows: int,
        norm_to: tuple[float, float] | T.Literal['targets'] | None,
        id: int,
    ):
        super().__init__()
        self.inputs: torch.Tensor | None = inputs[:n] if inputs is not None else None
        self.targets: torch.Tensor | None = targets[:n] if targets is not None else None
        self.initialized = False
        self.nrows = nrows
        self.id = id

        if isinstance(self.inputs, torch.Tensor):
            self.n = self.inputs.shape[0]
        else:
            self.n = n

        if norm_to == 'targets':
            #if self.targets is None: raise ValueError('norm_to = targets but targets are None')
            if self.targets is not None: norm_to = (self.targets.min().detach().cpu().item(), self.targets.max().detach().cpu().item())
            else: norm_to = None
        self.norm_to = norm_to

    @torch.no_grad
    def _normalize255(self, x:torch.Tensor):
        if self.norm_to is None: return _normalize(x, 0, 255)
        return ((x.clip(*self.norm_to) - self.norm_to[0]) / (self.norm_to[1] - self.norm_to[0])) * 255

    def _add(self, learner: "Learner", x: torch.Tensor | None, normalize = True):
        """Add (B, C, H, W) frame if it is not None. if more then 1 image in the batch, makes a grid.
        Clips to self norm_to and normalizes to 0, 255."""
        if x is not None:

            if x.ndim != 4: raise ValueError(f'x.shape = {x.shape}, must be (B, C, H, W)')

            if x.shape[0] == 1: x = x[0]
            else: x = make_grid(x, nrow=self.nrows)

            # normalize by clipping to self.norm_to and normalizing to 0, 255
            if normalize: x = self._normalize255(x)

            # add x as a tile
            learner.add_video_tile(x, id = self.id)

    def before_train_batch(self, learner: "Learner"):
        # make sure they are on correct device
        if not self.initialized:
            if 'add_video_tile' not in learner._events:
                raise ValueError('Please add the Renderer callback.')
            if isinstance(self.inputs, torch.Tensor): self.inputs = self.inputs.to(learner.device)
            if isinstance(self.targets, torch.Tensor): self.targets = self.targets.to(learner.device)
            self.initialized = True

        # concatenate inputs to batch, works with single tensor and tuple of tensor and target
        if self.inputs is not None:
            self.n = self.inputs.shape[0]
            if isinstance(learner.batch, torch.Tensor):
                learner.batch = torch.cat([learner.batch, self.inputs], dim=0)
            else:
                learner.batch = list(learner.batch)
                learner.batch[0] = torch.cat([learner.batch[0], self.inputs], dim=0)


class Render2DImageOutputsVideo(_BatchCatVideoCallback):
    order = 10

    def __init__(
        self,
        inputs,
        targets = None,
        n = None,
        activation: T.Optional[abc.Callable] = None,
        show_targets = True,
        show_inputs = True,
        nrows=2,
        norm_to: tuple[float, float] | T.Literal['targets'] | None = 'targets',
        id = 0,
    ):
        super().__init__(inputs, targets, n=n, nrows=nrows, norm_to=norm_to, id = id)
        self.show_targets = show_targets
        self.show_inputs = show_inputs
        self.activation = activation

    def after_forward(self, learner: "Learner"):
        if learner.status == 'train' and self.n is not None:

            # cut learner preds and get video preds
            learner.preds, video_preds = learner.preds[:-self.n], learner.preds[-self.n:]
            with torch.no_grad():
                if self.activation is not None: video_preds = self.activation(video_preds)

                # add inputs
                if self.show_inputs: self._add(learner, self.inputs)

                # add preds
                self._add(learner, video_preds)

                # add targets
                if self.show_targets: self._add(learner, self.targets)


class Render2DSegmentationVideo(_BatchCatVideoCallback):
    order = 10

    def __init__(
        self,
        inputs,
        targets: torch.Tensor | None=None,
        n = None,
        activation: T.Optional[abc.Callable] = None,
        nrows=2,
        show_inputs = True,
        show_raw_preds = True,
        show_binary_preds = True,
        show_binary_preds_overlay = True,
        show_targets = True,
        show_targets_overlay = True,
        show_error_overlay = True,
        binary_threshold = 0.5,
        alpha = 0.3,
        norm_to: tuple[float, float] | T.Literal['targets'] | None = None,
        id = 0
    ):
        """This saves:
        1. inputs
        2. raw preds
        3. binary preds
        4. binary preds overlaid
        5. raw preds channels if more than 1
        6. targets
        7. targets overlaid
        8. error overlaid
        """
        super().__init__(inputs, targets, n = n, nrows=nrows, norm_to=norm_to, id = id)
        self.activation = activation

        self.show_inputs = show_inputs
        self.show_raw_preds = show_raw_preds
        self.show_binary_preds = show_binary_preds
        self.show_binary_preds_overlay = show_binary_preds_overlay
        self.show_targets = show_targets
        self.show_targets_overlay = show_targets_overlay
        self.show_error_overlay = show_error_overlay
        self.binary_threshold = binary_threshold
        self.alpha = alpha

    def after_forward(self, learner: "Learner"):
        if learner.status == 'train' and self.n is not None:

            # cut learner preds and get video preds
            learner.preds, preds = learner.preds[:-self.n], learner.preds[-self.n:]
            with torch.no_grad():
                if self.activation is not None: preds = self.activation(preds)

                # add inputs
                if self.show_inputs: self._add(learner, self.inputs)

                # add raw preds
                if self.show_raw_preds: self._add(learner, preds)

                # add binary preds
                if self.show_binary_preds:
                    binary = preds.argmax(1)
                    binary = make_segmentation_overlay(binary.moveaxis(1, 0)).moveaxis(0,1)
                    self._add(learner, binary * 255, normalize = False)

                # add binary preds overlay
                if self.show_binary_preds_overlay and self.inputs is not None:
                    binary = preds.argmax(1)
                    preds_overlay = overlay_segmentation(self.inputs.moveaxis(0, -1), binary.moveaxis(0, -1), alpha=self.alpha).moveaxis(-1, 0)
                    self._add(learner, preds_overlay)

                # add targets
                if self.show_targets and self.targets is not None:
                    targets = self.targets
                    if targets.ndim == 4: targets = targets.argmax(1)
                    targets = make_segmentation_overlay(targets.moveaxis(0, -1)).moveaxis(-1, 0)
                    self._add(learner, targets * 255, normalize = False)

                # add targets overlay
                if self.show_binary_preds_overlay and self.targets is not None and self.inputs is not None:
                    targets = self.targets
                    if targets.ndim == 4: targets = targets.argmax(1)
                    targets_overlay = overlay_segmentation(self.inputs.moveaxis(0, -1), targets.moveaxis(0, -1), alpha=self.alpha).moveaxis(-1, 0)
                    self._add(learner, targets_overlay)

                # add error overlay
                if self.show_binary_preds_overlay and self.targets is not None and self.inputs is not None:
                    targets = self.targets
                    if targets.ndim == 4: targets = targets.argmax(1)
                    binary = preds.argmax(1)
                    targets_overlay = overlay_segmentation(self.inputs.moveaxis(0, -1), (binary == targets).moveaxis(0, -1), alpha=self.alpha).moveaxis(-1, 0)
                    self._add(learner, targets_overlay)




