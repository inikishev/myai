import typing as T

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from torchvision.utils import make_grid

from ..python_tools.f2f import (func2func, method2func, method2method,
                                method2method_return_override)
from ..torch_tools import make_segmentation_overlay
from ..torch_tools.conversion import (ensure_numpy_or_none_recursive,
                                      ensure_numpy_recursive,
                                      maybe_detach_cpu,
                                      maybe_detach_cpu_recursive)
from ..transforms import totensor
from ._norm import _normalize
from ._types import _FontSizes, _K_Collection, _K_Figure, _K_Line2D, _K_Text
from ._utils import _prepare_image_for_plotting

_PlotFwdRef: T.TypeAlias = "_Plot"
"""Forward reference to _Plot for the method2method_return_override decorator"""

class _Plot:
    def __init__(
        self,
        **kwargs: T.Unpack[_K_Figure],
    ):
        figure = kwargs.pop("figure", None)
        ax = kwargs.pop("ax", None)

        if "figsize" in kwargs and isinstance(kwargs["figsize"], (int, float)):
            kwargs["figsize"] = (kwargs["figsize"], kwargs["figsize"])

        kwargs.setdefault("layout", "constrained")

        if ax is None:
            if figure is None:
                fig_kwargs: dict = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {"projection", "polar", "label"}
                }
                figure = plt.figure(**fig_kwargs)
            ax_kwargs: dict = {
                k: v for k, v in kwargs.items() if k in {"projection", "polar", "label"}
            }
            ax = figure.add_subplot(**ax_kwargs)

        if figure is None: figure = ax.get_figure()
        if figure is None: raise ValueError("figure is None")

        self.figure: Figure = figure
        self.ax: Axes = ax

    def grid(
        self,
        major_alpha: float | None = 0.3,
        minor_alpha: float | None = 0.1,
        axis: T.Literal['both', 'x', 'y']="both",
        **kwargs: T.Unpack[_K_Line2D]
    ):
        k: dict[str, T.Any] = dict(kwargs)
        k.pop('alpha', None)
        if major_alpha is not None and major_alpha > 0: self.ax.grid(which = 'major', axis=axis, alpha=major_alpha, **k)
        if minor_alpha is not None and minor_alpha > 0: self.ax.grid(which = 'minor', axis=axis, alpha=minor_alpha, **k)
        return self

    def linechart(
        self,
        x=None,
        y=None,
        scalex=True,
        scaley=True,
        **kwargs: T.Unpack[_K_Line2D],
    ):
        if x is None and y is None: raise ValueError('No data to plot')
        if y is None:
            y = x
            x = None
        x = ensure_numpy_recursive(x)
        y = ensure_numpy_recursive(y)

        if y is None: raise ValueError("pylance")
        if x is None: self.ax.plot(y, scalex=scalex, scaley=scaley, **kwargs)
        else: self.ax.plot(x, y, scalex=scalex, scaley=scaley, **kwargs)
        return self

    def scatter(
        self,
        x,
        y,
        s=None,
        c=None,
        marker=None,
        vmin=None,
        vmax=None,
        alpha=None,
        plotnonfinite=False,
        **kwargs: T.Unpack[_K_Collection],
    ):
        x = ensure_numpy_recursive(x)
        y = ensure_numpy_recursive(y)
        c = maybe_detach_cpu_recursive(c)
        s = maybe_detach_cpu_recursive(s)

        loc = locals().copy()
        del loc["self"]
        kwargs = loc.pop("kwargs")
        loc.update(kwargs)

        self.ax.scatter(**loc)
        return self

    @method2method_return_override(Axes.imshow, _PlotFwdRef)
    def imshow(self,*args,**kwargs,):
        if len(args) >= 3:
            args = list(args)
            norm = args.pop(3)
        else: norm = kwargs.pop('norm', None)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'gray'


        if len(args) > 0:
            args = list(args)
            args[0] = _normalize(_prepare_image_for_plotting(ensure_numpy_recursive(args[0])), norm)

        elif 'X' in kwargs:
            kwargs['X'] = _normalize(_prepare_image_for_plotting(ensure_numpy_recursive(kwargs['X'])), norm)

        self.ax.imshow(*args, **kwargs)
        return self

    def segmentation(
        self,
        x: torch.Tensor | np.ndarray,
        alpha=0.3,
        bg_index = 0,
        colors=None,
        bg_alpha:float = 0.,
        **kwargs,
    ):
        x = totensor(maybe_detach_cpu_recursive(x), dtype=torch.float32).squeeze()
        # argmax if not argmaxed
        if x.ndim == 3:
            if x.shape[0] < x.shape[2]: x = x.argmax(0)
            else: x = x.argmax(-1)

        if x.ndim < 2: raise ValueError(f'Got x of shape {x.shape}')

        segm = make_segmentation_overlay(x, colors = colors, bg_index = bg_index) * 255
        segm = torch.cat([segm, torch.zeros_like(segm[0, None])], 0)
        segm[3] = torch.where(segm[:3].amax(0) > torch.tensor(0), torch.tensor(int(alpha*255)), torch.tensor(bg_alpha))

        self.ax.imshow(segm.permute(1,2,0).to(torch.uint8), **kwargs)
        return self


    def imshow_grid(
        self,
        x,
        nrows: int | None = None,
        ncols: int | None = None,
        padding: int = 2,
        value_range: tuple[int,int] | None = None,
        normalize: bool = True,
        scale_each: bool = False,
        pad_value: float = 0,
        **kwargs,
    ):
        x = torch.from_numpy(ensure_numpy_recursive(x))
        # add channel dim
        if x.ndim == 3: x = x.unsqueeze(1)
        # ensure channel first
        if x.shape[1] > x.shape[-1]: x = x.movedim(-1, 1)

        # distribute rows and cols
        if nrows is None:
            if ncols is None:
                ncols = len(x) ** 0.5
            nrows = len(x) / ncols # type:ignore

        # ensure rows are correct
        if nrows is None: raise ValueError('shut up pylance')
        nrows = round(nrows)
        if nrows < 1: nrows = 1

        # make the grid
        grid = make_grid(x, nrow=nrows, padding = padding, normalize=normalize, value_range=value_range, scale_each=scale_each, pad_value=pad_value)
        # this returns (C, H, W)

        return self.imshow(grid.moveaxis(0, -1), **kwargs,)

    def axtitle(
        self,
        label: T.Any,
        loc: T.Literal["center", "left", "right"] | None = None,
        pad: float | None = None,
        **kwargs: T.Unpack[_K_Text],
    ):
        self.ax.set_title(label = str(label)[:10000], loc = loc, pad = pad, **kwargs)
        return self

    def figtitle(self, t: T.Any, **kwargs: T.Unpack[_K_Text]):
        self.figure.suptitle(str(t)[:10000], **kwargs)
        return self

    def figsize(self, w: float | tuple[float, float] = (6.4, 4.8), h: float | None = None, forward: bool = True):
        self.figure.set_size_inches(w, h, forward)
        return self

    def xlabel(self, label: T.Any, **kwargs: T.Unpack[_K_Text]):
        self.ax.set_xlabel(str(label)[:10000], **kwargs)
        return self

    def ylabel(self, label: T.Any, **kwargs: T.Unpack[_K_Text]):
        self.ax.set_ylabel(str(label)[:10000], **kwargs)
        return self

    def axlabels(self, xlabel: T.Any, ylabel: T.Any, **kwargs: T.Unpack[_K_Text]):
        self.xlabel(xlabel, **kwargs)
        self.ylabel(ylabel, **kwargs)
        return self

    def legend(self, size: float | None=6, edgecolor=None, linewidth: float | None=3., frame_alpha = 0.3, prop = None):
        if prop is None: prop = {}
        if size is not None and 'size' not in prop: prop['size'] = size

        leg = self.ax.legend(prop=prop, edgecolor=edgecolor,)
        leg.get_frame().set_alpha(frame_alpha)

        if linewidth is not None:
            for line in leg.get_lines():
                line.set_linewidth(linewidth)

        return self

    @method2method(Axes.set_xlim)
    def xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)
        return self

    @method2method(Axes.set_ylim)
    def ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)
        return self

    @method2method(Axes.axis)
    def axis(self, *args, **kwargs):
        self.ax.axis(*args, **kwargs)
        return self

    def ticks(
        self,
        xmajor=True,
        ymajor=True,
        xminor: int | None | T.Literal["auto"] = "auto",
        yminor: int | None | T.Literal["auto"] = "auto",
    ):
        if xmajor: self.ax.xaxis.set_major_locator(AutoLocator())
        if ymajor: self.ax.yaxis.set_major_locator(AutoLocator())
        if xminor is not None: self.ax.xaxis.set_minor_locator(AutoMinorLocator(xminor)) # type:ignore
        if yminor is not None: self.ax.yaxis.set_minor_locator(AutoMinorLocator(yminor)) # type:ignore
        return self

    def tick_params(
        self,
        axis: T.Literal["x", "y", "both"] = "both",
        which: T.Literal["major", "minor", "both"] = "major",
        reset: bool = False,
        direction: T.Literal['in', 'out', 'inout'] | None = None,
        length: float | None = None,
        width: float | None = None,
        color = None,
        pad: float | None = None,
        labelsize: float | _FontSizes | None = None,
        labelcolor: T.Any | None = None,
        labelfontfamily: str | None = None,
        colors: T.Any | None = None,
        zorder: float | None = None,
        bottom: bool | None = None, top: bool | None = None, left: bool | None = None, right: bool | None = None,
        labelbottom: bool | None = None, labeltop: bool | None = None, labelleft: bool | None = None, labelright: bool | None = None,
        labelrotation: float | None = None,
        grid_color: T.Any | None = None,
        grid_alpha: float | None = None,
        grid_linewidth: float | None = None,
        grid_linestyle: str | None = None,
    ):
        loc: dict[str, T.Any] = locals().copy()
        del loc["self"]
        loc = {k:v for k,v in loc.items() if v is not None}
        self.ax.tick_params(**loc)
        return self

    def set_axis_off(self):
        self.ax.set_axis_off()
        return self

    def xscale(self, scale: str | T.Any):
        self.ax.set_xscale(scale)
        return self

    def yscale(self, scale: str | T.Any):
        self.ax.set_yscale(scale)
        return self
