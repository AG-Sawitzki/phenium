import os
from copy import deepcopy
from pathlib import Path, PurePath

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import tifffile
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from microfilm.microplot import microshow
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.typing import ArrayLike
from scipy import stats
from shapely import Point


def polygon_overlay(
    polygons: gpd.GeoDataFrame,
    img: str | os.PathLike | PurePath | ArrayLike,
    fig: Figure,
    ax: Axes,
    pxsize: float,
    **kwargs,
) -> tuple[Axes, Figure]:
    gdf = deepcopy(polygons)

    if (not isinstance(img, np.ndarray)) & (isinstance(img, str | PurePath)):
        img = tifffile.imread(img)

    assert type(img) is np.ndarray, f"Img has type {type(img)}."
    assert type(gdf) is gpd.GeoDataFrame, f"Polygons has {type(gdf)}."

    # img_norm = (img - img.min()) / (img.max() - img.min())
    img_norm = img

    # plt.style.use("./segmentstyle.mplstyle")
    colors = ["blue", "cyan", "magenta", "yellow"]

    fz = 24
    if img.ndim == 2:
        dimy, dimx = img.shape
        c = 1
    elif img.ndim == 3:
        c, dimy, dimx = img.shape
    else:
        dimy, dimx, *_ = img.shape


    # fig.set_frameon(False)
    fig.set_size_inches(dimy / 100, dimx / 100)
    ax.tick_params(axis="both", which="major", labelsize=fz)
    ax.set_xlabel("x_location in px", fontsize=fz)
    ax.set_ylabel("y_location in px", fontsize=fz)

    # ax.set_aspect('equal', 'box')

    size = dimx / 10
    length = np.round(size * pxsize, 0)

    asb = AnchoredSizeBar(
        ax.transData,
        size=size,
        label=f"{length} µm",
        loc="lower right",
        frameon=False,
        size_vertical=10 / pxsize,
        color="white",
        fontproperties=FontProperties(size=fz),
    )
    ax.add_artist(asb)

    ax.set_xlim(0, dimx)
    ax.set_ylim(dimy, 0)

    if img.ndim == 3:
        img_norm_m = img_norm.reshape((c, dimy, dimx))
        images = [img for img in img_norm_m]

        _microim = microshow(
            images=images,
            cmaps=colors[:c],
            ax=ax,
            limits=[img_norm_m.min(), img_norm_m.max()],
            show_axis=True,
        )
    else:
        images = [img]
        _microim = microshow(
            images=images,
            cmaps=colors[:c],
            ax=ax,
            limits=[img_norm.min(), img_norm.max()],
            show_axis=True,
        )

    gdf.boundary.plot(ax=ax, aspect="equal", color="white")
    fig.tight_layout()
    # fig.savefig(Path(output_path), bbox_inches="tight", pad_inches=0.0)
    return fig, ax


def scatter_doublets(
    stuff: dict[dict, ...],
    ax: Axes,
    fig: Figure,
    doublets: pl.DataFrame
) -> tuple[Axes, Figure]:
    for k, v in stuff.values():
        idxs = [int(idx) for idx in v]
        doublets_to_scatter = doublets[idxs]
        x = doublets_to_scatter.select('x').to_numpy()
        y = doublets_to_scatter.select('y').to_numpy()

        test_val = next(iter(v.values()))

        if len(test_val) > 1:
            if isinstance(test_val, Point):
                color = 'gray'
            else:
                color = 'green'
        else:
            color = 'red'

        ax.scatter(x, y, c=color)

        return tuple(ax, fig)


def plot_spatial_transcripts(
    table,
    ax,
    fig,
    pseudo_3d
):
    if pseudo_3d:
        columns = (cs.ends_with('_location'))
    else:
        columns = (cs.ends_with('_location') & ~cs.starts_with('z'))
    arr = table.select(columns).to_numpy()
    ax.scatter(arr[::100, 0], arr[::100, 1], c=(arr[:, 2] if pseudo_3d else 'b'))

    return fig, ax


def plot_transcripts_vs_z(
    table,
    ax,
    fig,
    axis,
    lingres,
):
    axis_arr = table.select(pl.col(f"{axis}_location")).to_numpy()
    z_arr = table.select(pl.col('z_location')).to_numpy()
    ax.scatter(axis_arr[::100], z_arr[::100], c='b')
    if lingres:
        gres = stats.linregress(axis_arr[::100], z_arr[::100])
        ax.plot(gres(axis_arr[::1000]), c='r')
    
    return fig, ax

