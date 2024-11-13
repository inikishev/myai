import typing as T
from collections.abc import Callable
from operator import attrgetter, methodcaller

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from itertools import zip_longest

from ..python_tools import Compose
from ..python_tools.f2f import (func2func, func2method, method2func,
                                method2method, method2method_return_override)
from ._plot import _Plot

_Fig: T.TypeAlias = "Fig"

class Fig:
    def __init__(self):
        self.plots: list[list[Callable[[_Plot], T.Any]]] = []
        self.titles: list[str | None] = []
        self.cur = 0
        self.fig_fns: list[Callable[[Fig], T.Any]] = []

    def __len__(self):
        return len(self.plots)

    def add(self, title: T.Optional[str | T.Any] = None):
        self.plots.append([])
        self.titles.append(str(title)[:10000] if title is not None else None)
        self.cur = len(self.plots) - 1
        return self

    def get(self, i: int):
        self.cur = i
        return self

    def _add_plot_func(self, name: str, *args, **kwargs):
        if len(self.plots) == self.cur: self.add()
        self.plots[self.cur].append(methodcaller(name, *args, **kwargs))
        return self

    def _add_figure_func(self, name: str, *args, **kwargs):
        self.fig_fns.append(Compose(attrgetter('figure'), methodcaller(name, *args, **kwargs)))
        return self

    @method2method_return_override(_Plot.linechart, _Fig)
    def linechart(self, *args, **kwargs): return self._add_plot_func("linechart", *args, **kwargs)

    @method2method_return_override(_Plot.scatter, _Fig)
    def scatter(self, *args, **kwargs): return self._add_plot_func("scatter", *args, **kwargs)

    @method2method_return_override(_Plot.imshow, _Fig)
    def imshow(self, *args, **kwargs): return self._add_plot_func("imshow", *args, **kwargs)

    @method2method_return_override(_Plot.imshow_grid, _Fig)
    def imshow_grid(self, *args, **kwargs): return self._add_plot_func("imshow_grid", *args, **kwargs)

    @method2method_return_override(_Plot.grid, _Fig)
    def grid(self, *args, **kwargs): return self._add_plot_func("grid", *args, **kwargs)

    @method2method_return_override(_Plot.axtitle, _Fig)
    def axtitle(self, *args, **kwargs): return self._add_plot_func("axtitle", *args, **kwargs)

    @method2method_return_override(_Plot.figtitle, _Fig)
    def figtitle(self, *args, **kwargs): return self._add_plot_func("figtitle", *args, **kwargs)

    @method2method_return_override(_Plot.figsize, _Fig)
    def figsize(self, *args, **kwargs): return self._add_plot_func("figsize", *args, **kwargs)

    @method2method_return_override(_Plot.xlabel, _Fig)
    def xlabel(self, *args, **kwargs): return self._add_plot_func("xlabel", *args, **kwargs)

    @method2method_return_override(_Plot.ylabel, _Fig)
    def ylabel(self, *args, **kwargs): return self._add_plot_func("ylabel", *args, **kwargs)

    @method2method_return_override(_Plot.axlabels, _Fig)
    def axlabels(self, *args, **kwargs): return self._add_plot_func("axlabels", *args, **kwargs)

    @method2method_return_override(_Plot.legend, _Fig)
    def legend(self, *args, **kwargs): return self._add_plot_func("legend", *args, **kwargs)

    @method2method_return_override(_Plot.xlim, _Fig)
    def xlim(self, *args, **kwargs): return self._add_plot_func("xlim", *args, **kwargs)

    @method2method_return_override(_Plot.ylim, _Fig)
    def ylim(self, *args, **kwargs): return self._add_plot_func("ylim", *args, **kwargs)

    @method2method_return_override(_Plot.axis, _Fig)
    def axis(self, *args, **kwargs): return self._add_plot_func("axis", *args, **kwargs)

    @method2method_return_override(_Plot.ticks, _Fig)
    def ticks(self, *args, **kwargs): return self._add_plot_func("ticks", *args, **kwargs)

    @method2method_return_override(_Plot.tick_params, _Fig)
    def tick_params(self, *args, **kwargs): return self._add_plot_func("tick_params", *args, **kwargs)

    @method2method_return_override(_Plot.set_axis_off, _Fig)
    def set_axis_off(self, *args, **kwargs): return self._add_plot_func("set_axis_off", *args, **kwargs)

    @method2method_return_override(_Plot.xscale, _Fig)
    def xscale(self, *args, **kwargs): return self._add_plot_func("xscale", *args, **kwargs)

    @method2method_return_override(_Plot.yscale, _Fig)
    def yscale(self, *args, **kwargs): return self._add_plot_func("yscale", *args, **kwargs)

    @method2method_return_override(_Plot.segmentation, _Fig)
    def segmentation(self, *args, **kwargs): return self._add_plot_func("segmentation", *args, **kwargs)

    def show(
        self,
        nrows: T.Optional[int | float] = None,
        ncols: T.Optional[int | float] = None,
        figure: "T.Optional[Figure | _Plot | Fig]" = None,
        figsize: float | tuple[float, float] | None = None,
        dpi: float | None = None,
        facecolor: T.Any | None = None,
        edgecolor: T.Any | None = None,
        frameon: bool = True,
        layout: T.Literal["constrained", "compressed", "tight", "none"] | None = 'compressed',
    ):
        # distribute rows and cols
        if ncols is None:
            if nrows is None:
                nrows = len(self.plots) ** 0.45
            ncols = len(self.plots) / nrows # type:ignore
        else:
            if ncols is None:
                ncols = len(self.plots) ** 0.55
            nrows = len(self.plots) / ncols # type:ignore

        # ensure rows and cols are correct
        if nrows is None or ncols is None: raise ValueError('shut up pylance')
        nrows = round(nrows)
        ncols = round(ncols)
        if nrows < 1: nrows = 1
        if ncols < 1: ncols = 1
        r = True
        while nrows * ncols < len(self.plots):
            if r: ncols += 1
            else: ncols += 1
            r = not r

        # create the figure if it is None
        if isinstance(figsize, (int,float)): figsize = (figsize, figsize)
        if isinstance(figure, (_Plot | Fig)): figure = figure.figure

        if figure is None:
            self.figure = plt.figure(
                figsize=figsize,
                dpi=dpi,
                facecolor=facecolor,
                edgecolor=edgecolor,
                frameon=frameon,
                layout=layout,
            )
        else:
            self.figure = figure

        # create axes
        _axes = self.figure.subplots(nrows = nrows, ncols = ncols)
        if isinstance(_axes, np.ndarray): self.axes = _axes.flatten()
        else: self.axes = [_axes]

        # plot
        for ax, label, fns in zip_longest(self.axes, self.titles, self.plots):
            plot = _Plot(ax=ax, figure=self.figure)

            if fns is not None:
                for fn in fns:
                    fn(plot)
            else:
                ax.set_axis_off()

            if label is not None:
                legend = True
                plot.axtitle(label)

    def clear(self):
        self.plots = []

    def savefig(self, path):
        self.figure.savefig(path, bbox_inches='tight', pad_inches=0)

    def close(self):
        plt.close(self.figure)

@method2func(Fig.linechart)
def linechart(*args, **kwargs) -> Fig: return Fig().add().linechart(*args, **kwargs)

@method2func(Fig.scatter)
def scatter(*args, **kwargs) -> Fig: return Fig().add().scatter(*args, **kwargs)

@method2func(Fig.imshow)
def imshow(*args, **kwargs) -> Fig: return Fig().add().imshow(*args, **kwargs)

def imshow_grid(images, labels = None, norm = 'norm', cmap = 'gray', **kwargs) -> Fig:
    if labels is None:
        labels = [None] * len(images)

    fig = Fig()
    for image, label in zip(images, labels):
        fig.add().imshow(image, norm = norm, cmap = cmap, **kwargs)
        if label is not None: fig.axtitle(label)

    fig.set_axis_off()
    return fig

