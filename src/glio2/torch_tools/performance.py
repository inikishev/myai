import torch
def performance_tweaks(cudnn_bench, onednn_fusion=True, detect_anomaly=False, checknan = False, autograd_profiler = False, emit_nvtx=False, gradcheck=False, gradgradcheck=False):
    """Make it go brrrrr

    :param cudnn_bench: _description_
    :param onednn_fusion: _description_, defaults to True
    :param detect_anomaly: _description_, defaults to False
    :param checknan: _description_, defaults to False
    :param autograd_profiler: _description_, defaults to False
    :param emit_nvtx: _description_, defaults to False
    :param gradcheck: _description_, defaults to False
    :param gradgradcheck: _description_, defaults to False
    """
    if cudnn_bench is not None: torch.backends.cudnn.benchmark = cudnn_bench
    if onednn_fusion is not None: torch.jit.enable_onednn_fusion(onednn_fusion)
    if detect_anomaly is not None: torch.autograd.set_detect_anomaly(detect_anomaly, checknan) # type:ignore
    if autograd_profiler is not None: torch.autograd.profiler.profile(autograd_profiler) # type:ignore
    if emit_nvtx is not None: torch.autograd.profiler.emit_nvtx(emit_nvtx) # type:ignore