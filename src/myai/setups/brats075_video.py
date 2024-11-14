import torch
from glio.jupyter_tools import show_slices_arr
from monai.losses import DiceFocalLoss  # type:ignore

from ..datasets.mrislicer_utils import ReferenceSlice, make_ds
from ..learner import CB, Learner
from ..loaders.nifti import niiread
from ..python_tools import find_file_containing
from ..torch_tools import crop_around, overlay_segmentation

def get_learner(
    dltest,
    model,
    opt,
    loss,
    sched,
    reference_slice: ReferenceSlice,
    around: int,
    window_size = (96, 96),
    every: int | None = None,
    before_fit = False,
    after_fit = False,
    max_steps = 1000,
    fps = 30,
    extra_cbs = ()
) -> Learner:

    ref_image, ref_seg = reference_slice.get((window_size), around)
    callbacks = [
        CB.test_epoch(dltest, every = every, before_fit = before_fit, after_fit = after_fit),
        CB.StopOnStep(max_steps),
        CB.LogTime(),
        CB.LogLoss(),
        CB.Accelerate(),
        CB.Dice(['bg', 'necrosis', 'edema', 'tumor', 'resection'], train_step=8, bg_index=0),
        CB.PerformanceTweaks(True),
        CB.FastProgress(('train loss', 'test loss', 'train dice mean', 'test dice mean'), 0.2, 30, ybounds = (0, 1)),
        CB.Renderer(fps=fps, nrows=4),
        CB.Render2DSegmentationVideo(
            inputs = ref_image.unsqueeze(0).to(torch.float32),
            targets = ref_seg.unsqueeze(0),
            n = 1,
            nrows = 2,
            activation=None,
            inputs_grid=True,
            overlay_channel=0,
        ),
    ]
    callbacks.extend(extra_cbs)
    learner = Learner(
        callbacks=callbacks,
        model=model,
        loss_fn=loss,
        optimizer=opt,
        scheduler=sched,
        main_metric = 'test dice',
    )
    return learner
