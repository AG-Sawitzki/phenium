import os
from copy import deepcopy
from pathlib import PurePath

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


def polygon_overlay(
    polygons: gpd.GeoDataFrame,
    img: str | os.PathLike | PurePath | ArrayLike,
    pxsize: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Overlay a segmentation mask in a geopandas GeoDataFrame ontop of an image with 1 to 4 channels.

    Parameter
    ---------
        polygons: gpd.GeoDataFrame
            A GeoDataFrame with a geometry column of cell-boundaries.
        img: str, PurePath or ArrayLike
            The image underlayed the polygons.
        pxsize: float or None
            Pixelsize of the image in x,y direction. Optional.
        ax: Axes or None
            Axes onto which to plot. Optional.

    Return
    ------
        out: Figure, Axes
            Figure & Axes with the img and polygons ploted.
    """
    if ax is None:
        fig, ax = plt.sublots()
    else:
        fig = ax.get_figure()
    gdf = deepcopy(polygons)

    if (not isinstance(img, np.ndarray)) & (isinstance(img, str | PurePath)):
        img = tifffile.imread(img)

    assert type(img) is np.ndarray, f"Img has type {type(img)}."
    assert type(gdf) is gpd.GeoDataFrame, f"Polygons has {type(gdf)}."

    img_norm = img

    colors = ["blue", "cyan", "magenta", "yellow"]

    fz = 24
    if img.ndim == 2:
        dimy, dimx = img.shape
        c = 1
    elif img.ndim == 3:
        c, dimy, dimx = img.shape
    else:
        dimy, dimx, *_ = img.shape

    fig.set_frameon(False)
    fig.set_size_inches(dimy / 100, dimx / 100)
    ax.tick_params(axis="both", which="major", labelsize=fz)
    ax.set_xlabel("x_location in px", fontsize=fz)
    ax.set_ylabel("y_location in px", fontsize=fz)

    ax.set_aspect('equal', 'box')

    size = dimx / 10
    if pxsize:
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

    gdf.boundary.plot(color="white", ax=ax, aspect="equal")
    fig.tight_layout()
    return ax


def scatter_doublets(
    stuff: dict[dict, ...],
    doublets: pl.DataFrame,
    ax: Axes | None = None
) -> Axes:
    if ax is None:
        _, ax = plt.sublots()
    for k, v in stuff.items():
        idxs = [int(idx) for idx in v]
        doublets_to_scatter = doublets[idxs]
        x = doublets_to_scatter.select('x').to_numpy().ravel()
        y = doublets_to_scatter.select('y').to_numpy().ravel()

        if k == 'single_cell':
            color = 'yellow'
        if k == 'double_cell':
            color = 'green'
        if k == 'no_cell':
            color = 'red'

        ax.scatter(x, y, c=color, s=500)

        return ax


def spatial_transcripts(
    table: pl.DataFrame,
    pseudo_3d: bool = False,
    ax: Axes | None = None
) -> Axes:
    """
    Scatter plot of every 100th transcript. Optionally colorcodes the z-Position.

    Parameter
    ---------
        table: polars.DataFrame
            DataFrame with columns named `{axis}_location` or `{axis}`.
            At least for x- and y-position.
        pseudo_3d: bool
            If `True` color code the transcripts by z-position.
            Default `False`.
        ax: Axes
            Axes to plot the transcripts on. Optional.
    
    Returns
    -------
        out: Axes
            Axes with scatter plot and a colorbar if `pseudo_3d=True`.
    """
    if ax is None:
        fig, ax = plt.sublots()
    else:
        fig = ax.get_figure()
    if pseudo_3d:
        columns = (
            (cs.ends_with("_location"))
            | (cs.starts_with("x")
                | cs.starts_with("y")
                | cs.starts_with('z'))
        )
    else:
        columns = (
            (cs.ends_with("_location") & ~cs.starts_with("z"))
            | (cs.starts_with("x") | cs.starts_with("y"))
        )
    arr = table.select(columns).to_numpy()
    scttr = ax.scatter(
        arr[::100, 0], arr[::100, 1], c=(arr[::100, 2] if pseudo_3d else "#009AA9"), s=0.5
    )
    if pseudo_3d:
        fig.colorbar(scttr, ax=ax, orientation="vertical")
    ax.set_aspect('equal')
    ax.set_xlabel('x_location in px')
    ax.set_ylabel('y_location in px')

    return ax


def transcripts_vs_z(
    table: pl.DataFrame,
    axis: str,
    lingres: bool = False,
    color_fovs: bool = False,
    ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """
    Scatter plot of the z-position of transcripts along `axis`.
    Can be color-coded by fov (Xenium-specific) or with a linear-regression.

    Parameters
    ----------
        table: polars.DataFrame
            DataFrame with columns named `{axis}_location` and `z_location`.
            If `color_fovs=True` then also `fov_name`-column.
        axis: str
            Name of the axis. 'x' or 'y'.
        lingres: bool
            If `True` add a linear-regression line of the transcript distribution.
            Default `False`.
        color_fovs: bool
            If `True` plot each fov in a separate color.
            Default `False`.
        ax: Axes
            Axes to plot on. Optional.
    
    Returns
    -------
        out: Axes
            Axes with scatter plot added.
    """
    if ax is None:
        fig, ax = plt.sublots()
    else:
        fig = ax.get_figure()

    if lingres & color_fovs:
        raise NotImplementedError('Cannot add regression line and color each fov differently.')

    if color_fovs:
        fov_names = list(set(table['fov_names']))
        for fov in fov_names:
            table_ = table.filter(pl.col('fov_name')==fov)
            z_arr = table_.select(pl.col('z_location')).to_numpy()[::100].ravel()
            axis_arr = table_.select(pl.col(f'{axis}_location')).to_numpy()[::100].ravel()
            ax.scatter(axis_arr, z_arr, s=0.5)
    else:
        axis_arr = table.select(pl.col(f"{axis}_location")).to_numpy()[::100].ravel()
        z_arr = table.select(pl.col("z_location")).to_numpy()[::100].ravel()
        ax.scatter(axis_arr, z_arr, c="#009AA9", s=0.5)
    
    if lingres:
        gres = stats.linregress(axis_arr, z_arr)
        ax.plot(axis_arr, (gres.intercept + gres.slope*axis_arr), c="#EA5451")
        ax.set_title(
            f"Distribution of along {axis}-axis."
            f"Slope: {gres.slope}; Intercept: {gres.intercept}."
        )
    ax.set_xlabel(f'{axis}_location in px')
    ax.set_ylabel('z_location in µm')

    return fig, ax


def z_distribution(
    table: pl.DataFrame,
    cutoff: bool = False,
    ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """
    Graph-plot of the z-distribution with a kernel-density-estimation.

    Parameters
    ----------
        table: polars.DataFrame
            DataFrame with columns named `{axis}_location` and `z_location`.
            If `color_fovs=True` then also `fov_name`-column.
        cutoff: bool
            Add a line indicating the minimum and maximum including 95% of data.
            Default `False`.
        ax: Axes
            Axes to plot on. Optional.

    Returns
    -------
        out: Axes
            Axes with distribution plot.
    """
    if ax is None:
        fig, ax = plt.sublots()
    else:
        fig = ax.get_figure()

    z_kde = stats.gaussian_kde(
        table.select(pl.col('z_location').to_numpy().ravel())
    )

    z_min = table.select(pl.min('z_location'))
    z_max = table.select(pl.max('z_location'))

    z_space = np.linspace(z_min, z_max, 100).ravel()

    z_density = z_kde(z_space)

    ax.plot(z_space, z_density, c="#009AA9")
    ax.set_xlabel('z_location in µm')
    ax.set_ylabel('fraction of transcripts')

    if cutoff:
        z_density_argsort = np.argsort(z_density)[::-1]
        z_min = 0
        z_max = 0
        density_sum = 0
        for z_idx in z_density_argsort:
            dens = z_density[z_idx]
            z_val = z_space[z_idx]
            density_sum += dens
            if density_sum / sum(z_density) > 0.95:
                break
            z_min = min(z_min, z_val)
            z_max = max(z_max, z_val)
            if z_min == 0:
                z_min = z_val
            if z_max == 0:
                z_max = z_val

        ax.vlines([z_min, z_max], z_density.min(), z_density.max(), colors="#EA5451")
        ax.set_title(f'Distribution of along Z-axis. Cutoff at {z_min} and {z_max}.')

    return ax
