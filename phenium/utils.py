from phenium.exceptions import NotImplemented

import os
from math import sqrt
from pathlib import Path
from itertools import product
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import xml.etree.ElementTree as ET

import numpy as np
import polars as pl
import geopandas as gpd
import scanpy as sc
import rapids_singlecell as rsc
from shapely import Polygon
# from ants import registration, apply_transforms
# from scipy.io import loadmat
import cupyx
import scipy
import cv2
import ovrlpy
import tifffile

from typing import Any, Union, List, Tuple, Dict, overload
from pathlib import PosixPath
from numpy.typing import ArrayLike


def depth(
    d: dict
) -> int:
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0


def to_cli(
    tup: Tuple[str, bool | int | float | str],
    prefix: str = '',
    suffix: str = ''
) -> str:
    """
    A argument-value pair is written as a Command-Line Interface argument.

    Parameters
    ----------
        tup: tuple
            A tuple of two entries. First the argument, second the value.
        prefix: str
            A string append before the argument.
        suffix: str
            A string append following the argument.

    Retruns
    -------
        out: str
            A CLI-argument
            --[prefix][arg][suffix] [value]
    """
    arg, value = tup
    if isinstance(value, bool):
        if value:
            return f'--{prefix}{arg}{suffix}'
        else:
            return ''
    elif isinstance(value, (int, float, str)):
        return f'--{prefix}{arg}{suffix} {value}'
    else:
        return ''


def find_rois(
    shape_org: tuple,
    img_subres: ArrayLike,
):
    """
    Sort the contours by area.

    Parameters:
    ----------
        shape_org : tuple
            Max resolution of img.
        img_subres : ArrayLike
            Lowest subresolution of image.

    Returns
    ----------
        out : list
            Contours of significant size.
    """
    c, y, x = shape_org

    subres = img_subres[0, ...]
    subres_normed = subres/np.max(subres)
    subres_U8I = np.uint8(subres_normed*255)
    subres_dilated = cv2.dilate(
        subres_U8I,
        np.ones((3, 3)),
        iterations=3
    )
    _, subres_binary = cv2.threshold(
        subres_dilated,
        int(np.average(subres_dilated)), 255, 0
    )
    contours, _ = cv2.findContours(
        subres_binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []
    scores = []
    for c in contours:
        (x, y), (w, h), a = cv2.minAreaRect(c)
        dim = w if h < w else h
        bboxes.append([x,y,dim,dim])
        scores.append(w*h)

    scores_normed = list(map(lambda x: x/max(scores), scores))

    ids_to_keep = cv2.dnn.NMSBoxes(
        bboxes,
        scores=scores_normed,
        score_threshold=0.1,
        nms_threshold=0.0,
    )

    bboxes_to_keep = np.array(bboxes)[ids_to_keep]
    bboxes_sorted = sorted(bboxes_to_keep, key=lambda k: [int(k[1]/100), k[0]])

    return bboxes_sorted


@overload
def bboxes_to_sections(
    bboxes: List[ArrayLike],
    shape_org: Tuple,
    subres: Tuple,
) -> dict:
    pass
@overload
def bboxes_to_sections(
    bboxes: List[ArrayLike],
    shape_org: Tuple,
    subres: ArrayLike,
) -> Tuple[dict, ArrayLike]:
    pass
def bboxes_to_sections(
    bboxes,
    shape_org,
    subres,
):
    """

    """
    c, y, x = shape_org
    if isinstance(subres, tuple):
        c_s, y_s, x_s = subres
    else:
        marked_img = np.uint8(subres[0, ...]).copy()
        y_s, x_s = marked_img.shape

    rf_x = int(x/x_s)
    rf_y = int(y/y_s)

    sections = dict()

    for section, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        sections[section] = {
            'x': [int((x-w/2)*rf_x), int((x+w/2)*rf_x)],
            'y': [int((y-h/2)*rf_y), int((y+h/2)*rf_y)]
        }
        if not isinstance(subres, tuple):
            cv2.rectangle(
                marked_img, (int((x-w/2)), int(y-h/2)), (int(x+w/2), int(y+h/2)),
                (255, 255, 255), 2
            )
            cv2.putText(
                img=marked_img,
                text=str(section),
                org=(int(x-(w/4)), int(y)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )

    if isinstance(subres, tuple):
        return sections
    else:
        return sections, marked_img


def filter_by_location(
    table: Union[pl.DataFrame, gpd.GeoDataFrame],
    coords: dict,
    z_bounds: tuple = None,
    pixelsize: float = None,
) -> Union[pl.DataFrame, gpd.GeoDataFrame]:
    """
    Keep entries in `table` that have coordinates in `coords`.
    `coords` and `table` do not have the same units add `pixelsize`.
    `pixelsize` is applied to `table`, not `coords`.

    Parameters
    ----------
        table : polars or geopandas dataframe
            Contains the coordinates to filter.
            Column with `vertex_` or `_location` for polars.
            `geometry` column in GeoDataFrame.
        coords : dict
            Dictionary containing min. x,y and max. x,y.
            For an example, see `gt_section.json`
        z_bounds : tuple, optional
            Tuple of lower and upper z-Axis boundaries.
            Only for polars dataframe with "z_location" column.
        pixelsize : float, optional
            Size of a pixel in X & Y. Used to scale the values in `coords`.

    Returns
    -------
        out : polars or geopandas dataframe
            Same table as input, excluding all entries with coordinates not in
                `coords` 
                & `z_bounds`.
    """
    assert depth(coords) == 1, \
        f'`coords` incorrect: {coords}\n Consult example .json!'
    if isinstance(table, pl.DataFrame):
        for axis, extremes in coords.items():
            var_min, var_max = extremes
            # print(var_min, '\n', var_max, '\n', extremes)

            if pixelsize:
                var_min = float(var_min * pixelsize)
                var_max = float(var_max * pixelsize)
            if f'{axis}_location' in table.columns:
                column = f'{axis}_location'
            elif f'vertex_{axis}' in table.columns:
                column = f'vertex_{axis}'
            else:
                column = 'x'

            expr = ((pl.col(column)).is_between(
                var_min, var_max, closed='both'
            ))
            table = table.filter(expr)
            print(axis, extremes)
            print(table.shape)
        if z_bounds != None and 'z_location' in table.columns:
            z_low, z_high = z_bounds
            expr = ((pl.col('z_location')).is_between(
                z_low, z_high, closed='both'
            ))
            table = table.filter(expr)
            print(z_low, z_high)
            print(table.shape)
    elif isinstance(table, gpd.GeoDataFrame):
        x_min, x_max = coords['x']
        y_min, y_max = coords['y']

        if pixelsize:
            x_min = float(x_min * pixelsize)
            x_max = float(x_max * pixelsize)
            y_min = float(y_min * pixelsize)
            y_max = float(y_max * pixelsize)

        polygon = Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_min, y_max), (x_max, y_max),
            (x_min, y_min),
        ])
        # if pixelsize:
        #     geometry = table['geometry'].transform(
        #         lambda x: (x * np.array([
        #             1/pixelsize, 1/pixelsize
        #         ])).round(0)
        #     )
        #     table['geometry'] = geometry
        check = table['geometry'].within(polygon)
        table = table[check]
        if table.index.name in table.columns:
            table.drop(table.index.name, axis=1, inplace=True)
    else:
        raise NotImplementedError(
            f'Not Implemented to deal with {type(table)}.'
        )

    return table


def boundaries_to_mask(
    table: pl.DataFrame,
    shape: tuple
) -> Union[ArrayLike, Dict]:
    """

    """
    data = dict()
    masks = np.zeros(shape)

    for cell_id in set(table['cell_id']):
        x = table[table['cell_id'] == cell_id]['vertex_x']
        y = table[table['cell_id'] == cell_id]['vertex_y']
        if len(x) >= 4 or len(y) >= 4:

            geometry = Polygon(zip(x,y))
            data.update({
                cell_id : geometry
            })
            mask = np.zeros_like(masks, np.uint8)
            cv2.fillPoly(mask, np.int32([geometry]), (1,1,1))
            masks[mask > 0] = cell_id
    
    return masks, data


def transcripts_for_ovrlpy(
    machine: str,
    transcripts: Union[pl.DataFrame, os.PathLike, PosixPath],
) -> pl.DataFrame:
    """
    Rename the columns and filter the genes. Prepares the table for `ovrlpy`.
    """
    if machine == 'xenium':
        if isinstance(transcripts, (os.PathLike, str)):
            transcripts = ovrlpy.io.read_Xenium(
                filepath=transcripts,
                min_qv=None,
                remove_features=ovrlpy.io.XENIUM_CTRLS,
                n_threads=cpu_count(),
            )
        elif isinstance(transcripts, pl.DataFrame):
            transcripts = transcripts.rename(
                ovrlpy.io._10X_GENOMICS_COLUMNS
            )
            transcripts = ovrlpy.io._filter_genes(
                transcripts, ovrlpy.io.XENIUM_CTRLS
            )
        else:
            raise TypeError(
                'Only accepts paths and polars.DataFrame,'
                f' not {type(transcripts)}.'
            )
    else:
        raise NotImplementedError(
            f'Not Implemented to deal with `{machine}`.'
        )

    return transcripts


def get_pixel_size(
    img_path: Union[str, os.PathLike, PosixPath]
) -> Tuple[float, float, float]:
    """
    Get pixel size from xml metadata of (OME)Tiff.
    Returns `-1` for nonexisting Physical Sizes.

    Parameters
    ----------
        img_path : str, os.PathLike, PosixPath
            Path to a *tiff img.

    Returns
    -------
        out : tuple
            Tuple of size 3 with size of a pixel in x,y,z direction.
    """
    with tifffile.TiffFile(img_path) as tiff:
        tags = tiff.pages[0].tags

    root = ET.fromstring(tags[270].value)
    pixels = root.find('.//*[@PhysicalSizeX]')
    if isinstance(pixels, ET.Element):
        px_size_X = pixels.get('PhysicalSizeX')
        px_size_Y = pixels.get('PhysicalSizeY')
        px_size_Z = pixels.get('PhysicalSizeZ', default=-1)
    elif isinstance(pixels, type(None)):
        px_size_X = tags[282].value[1]/tags[282].value[0]*1e4
        px_size_Y = tags[283].value[1]/tags[283].value[0]*1e4
        px_size_Z = -1
    else:
        raise ValueError('Cannot find the pixelsize.')

    return float(px_size_X), float(px_size_Y), float(px_size_Z)


def run_leiden(
    adata,
    gpu=False,
):
    """
    Prepare for and then run Leiden on an anndata object.

    Parameters
    ----------
        adata : anndata
            contains genes, counts and cells
        gpu : bool, optional
            `True` if a gpu is avaiable and should be used
            `False` if not. Default is `False`
    
    Returns
    -------
        out : anndata
            anndata with "umap", "leiden" in obs(?)
    """
    sc.pp.calculate_qc_metrics(
        adata,
        percent_top=(25, 75, 150),
        inplace=True,
        log1p=True
    )

    sc.pp.filter_cells(adata, min_genes=3)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.normalize_total(adata, 1e4, max_fraction=0.05)
    sc.pp.log1p(adata)

    sc.tl.pca(adata, n_comps=100)

    if gpu:
        rsc.get.anndata_to_GPU(adata, convert_all=True)
        rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
        rsc.tl.umap(adata)
        rsc.tl.leiden(adata, resolution=0.1)
        rsc.get.anndata_to_CPU(adata, convert_all=True)
    else:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=0.2)

    return adata