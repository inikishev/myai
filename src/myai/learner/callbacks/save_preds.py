import os
import typing as T
from collections import abc

import torch

from ...event_model import Callback, ConditionalCallback
from ...plt_tools.fig import Fig
from ...torch_tools import make_segmentation_overlay, overlay_segmentation


if T.TYPE_CHECKING:
    from ..learner import Learner


class Save2DImagePreds(ConditionalCallback):
    order = 10

    def __init__(
        self,
        inputs,
        targets = None,
        labels = None,
        n = None,
        activation: T.Optional[abc.Callable] = None,
        show_targets = True,
        show_grads = True,
        show_inputs = True,
        dir="preds",
        root="runs",
        nrows=None,
        ncols=None,
        figsize=16,
    ):
        super().__init__()
        self.inputs = inputs[:n]
        self.targets = targets[:n] if targets is not None else None
        self.dir = dir
        self.root = root
        self.show_targets = show_targets
        self.show_inputs = show_inputs
        self.show_grads = show_grads
        self.activation = activation
        self.nrows, self.ncols = nrows, ncols
        self.figsize = figsize

        if labels is None: labels = list(range(len(inputs)))
        self.labels = labels

    def __call__(self, learner: "Learner"):
        # dir to save to
        dir = os.path.join(learner.get_learner_dir(self.root), self.dir)
        if not os.path.exists(dir):
            os.mkdir(dir)

        preds = learner.inference(self.inputs, enable_grad=self.show_grads).requires_grad_(True)
        if self.activation is not None:
                preds = self.activation(preds)

        # evaluate loss and the gradients
        if self.show_grads:
            with torch.enable_grad():
                t = self.targets
                if isinstance(t, torch.Tensor): t = t.to(learner.device)

                loss = learner.get_loss(preds, t)
                grads = torch.autograd.grad(loss, preds)[0]
        else: grads = [None] * len(preds)

        # plotting
        fig = Fig()
        for input, output, grad, label in zip(self.inputs, preds, grads, self.labels):
            if self.show_inputs: fig.add(f'input {label}').imshow(input).axis('off')
            fig.add(f'output {label}').imshow(output).axis('off')

            if self.targets is not None and self.show_targets:
                fig.add(f'target {label}').imshow(self.targets[label]).axis('off')

            if grad is not None:
                fig.add(f'output grad {label}').imshow(grad).axis('off')

        fig.show(self.nrows, self.ncols, figsize = self.figsize)
        fig.savefig(os.path.join(dir, f'preds e{learner.total_epochs} s{learner.num_forwards}.png'))
        fig.close()


class Save2DSegmentationPreds(ConditionalCallback):
    order = 10

    def __init__(
        self,
        inputs,
        targets=None,
        labels = None,
        dir="preds",
        root="runs",
        activation: T.Optional[abc.Callable] = None,
        nrows=None,
        ncols=None,
        figsize=24,
        binary_threshold = 0.5,
        alpha = 0.3,
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

        :param inputs: The input batch.
        :param targets: Sequence of targets for each input, defaults to None
        :param labels: Labels for each input, defaults to None
        :param dir: directory in root, defaults to "preds"
        :param root: root learner directory, defaults to "runs"
        :param activation: activation to apply to preds, defaults to None
        :param nrows: rows in saved figure, defaults to None
        :param ncols: cols in saved figure, defaults to None
        :param figsize: Size of saved figure, defaults to 24
        :param binary_threshold: binary threshold for binarizing single channel preds, defaults to 0.5
        :param alpha: Alpha of segmentation overlay, defaults to 0.3
        """
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.dir = dir
        self.root = root
        self.activation = activation
        self.nrows, self.ncols = nrows, ncols
        self.figsize = figsize
        self.binary_threshold = binary_threshold
        self.alpha = alpha
        if labels is None: labels = list(range(len(inputs)))
        self.labels = labels

    def __call__(self, learner: "Learner"):
        # dir to save to
        dir = os.path.join(learner.get_learner_dir(self.root), self.dir)
        if not os.path.exists(dir):
            os.mkdir(dir)

        # inference
        preds = learner.inference(self.inputs)
        if self.activation is not None:
            preds = self.activation(preds)

        # plotting
        fig = Fig()
        for i, (input, output, label) in enumerate(zip(self.inputs, preds, self.labels)):
            fig.add(f'input {label}').imshow(input).axis('off')
            fig.add(f'raw output {label}').imshow(output).axis('off')

            # output is converted to binary segmentation
            if output.shape[0] == 0:
                binarized = torch.where(output > self.binary_threshold, 1, 0)
            else:
                # we also add a grid of all channel outputs
                fig.add(f'raw output channels {label}').imshow_grid(output, normalize=True, scale_each=True).axis('off')
                binarized = output.argmax(0)
            fig.add(f'output {label}').segmentation(binarized, alpha = 1, bg_alpha=1).axis('off')
            fig.add(f'output {label} overlay').imshow(input).segmentation(binarized, alpha = self.alpha).axis('off')

            # targets
            if self.targets is not None:
                target = self.targets[i]
                # targets are converted to binary segmentation
                if target.ndim == 3:
                    if target.shape[0] == 0:
                        bin_target = torch.where(target > self.binary_threshold, 1, 0)
                    else:
                        bin_target = target.argmax(0)
                else: bin_target = target

                # we add raw targets, overlaid on input, and error
                fig.add(f'target {label}').imshow(bin_target).axis('off')
                fig.add(f'target {label} overlay').imshow(input).segmentation(bin_target, alpha = self.alpha).axis('off')
                fig.add(f'error {label} overlay').imshow(input).segmentation(bin_target != binarized, alpha = self.alpha).axis('off')

        fig.show(self.nrows, self.ncols, figsize = self.figsize)
        fig.savefig(os.path.join(dir, f'preds e{learner.total_epochs} s{learner.num_forwards}.png'))



