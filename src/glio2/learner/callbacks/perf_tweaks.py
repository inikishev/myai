from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from ...torch_tools import performance_tweaks
from ...event_model import Callback

if TYPE_CHECKING:
    from ..learner import Learner

class PerformanceTweaks(Callback):
    order = -1000
    def __init__(self, cudnn_bench, onednn_fusion=True, detect_anomaly=False, checknan = False, autograd_profiler = False, emit_nvtx=False, gradcheck=False, gradgradcheck=False):
        super().__init__()
        self.cudnn_bench = cudnn_bench
        self.onednn_fusion = onednn_fusion
        self.detect_anomaly = detect_anomaly
        self.checknan = checknan
        self.autograd_profiler = autograd_profiler
        self.emit_nvtx = emit_nvtx
        self.gradcheck = gradcheck
        self.gradgradcheck = gradgradcheck


    def enter(self, learner: "Learner"):
        performance_tweaks(
            cudnn_bench=self.cudnn_bench,
            onednn_fusion=self.onednn_fusion,
            detect_anomaly=self.detect_anomaly,
            checknan=self.checknan,
            autograd_profiler=self.autograd_profiler,
            emit_nvtx=self.emit_nvtx,
            gradcheck=self.gradcheck,
            gradgradcheck=self.gradgradcheck
        )
