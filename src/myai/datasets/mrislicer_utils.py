from collections import abc

import joblib
import torch
from mrid.training.mri_slicer import MRISlicer

from ..python_tools import reduce_dim


def load_old_mrislicer_dataset(
    path, num_classes: int, around: int, any_prob: float = 0.1, warn_empty: bool = True
) -> list[MRISlicer]:
    """Load dataset created with old glio MRISlicer"""
    old_format = joblib.load(path)
    dataset = []
    for sample in old_format:
        dataset.append(
            MRISlicer(
                sample.tensor,
                sample.seg,
                num_classes=num_classes,
                around=around,
                any_prob=any_prob,
                warn_empty=warn_empty,
            )
        )

    return dataset

def load_mrislicer_dataset(path) -> list[MRISlicer]:
    """Load dataset created with new glio MRISlicer"""
    return joblib.load(path)

def get_seg_slices(dataset: list[MRISlicer] | str) -> list[abc.Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a dataset that contain segmentation + `any_prob` * 100 % objects that return a random slice."""
    if isinstance(dataset, str): dataset = load_mrislicer_dataset(dataset)
    ds = reduce_dim([i.get_all_seg_slice_callables() for i in dataset])
    random_slices = reduce_dim([i.get_anyp_random_slice_callables() for i in dataset])
    ds.extend(random_slices)
    return ds

def get_all_slices(dataset: list[MRISlicer] | str) -> list[abc.Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a dataset"""
    if isinstance(dataset, str): dataset = load_mrislicer_dataset(dataset)
    ds = reduce_dim([i.get_all_slice_callables() for i in dataset])
    return ds

