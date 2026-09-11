from phenium.exceptions import NotImplemented

import os
from math import sqrt
from pathlib import Path
from itertools import product
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
import polars as pl
import geopandas as gpd
from shapely import Polygon
import cv2
import tifffile
import ovrlpy

from typing import Any, Union, List, Tuple
from pathlib import PosixPath
from numpy.typing import ArrayLike

def depth(
    d: dict
) -> int:
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0


def to_cli(tup: tuple, prefix = '', suffix = ''):
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
) -> Union[list, ArrayLike]:
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
        out : ArrayLike or list
            Contours of significant size.
    """
    z, y, x = shape_org

    subres_centre = np.uint8(img_subres[z//2])

    subres_dilated = cv2.dilate(
        subres_centre,
        np.ones((2, 2)),
        iterations=5
    )
    _, subres_binary = cv2.threshold(
        subres_dilated,
        127, 255, 0
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
        bboxes.append([x,y,w,h])
        scores.append(w*h)

    scores_normed = list(map(lambda x: x/max(scores), scores))

    ids_to_keep = cv2.dnn.NMSBoxes(
        bboxes,
        scores=scores_normed,
        score_threshold=0.1,
        nms_threshold=0.0,
    )

    bboxes_to_keep = np.array(bboxes)[ids_to_keep]
    bboxes_sorted = sorted(bboxes_to_keep, key=lambda k: [k[1], k[0]])

    return bboxes_sorted


def bboxes_to_sections(
    bboxes: List[ArrayLike],
    shape_org: Tuple,
    shape_subres: Tuple,
) -> dict:
    z, y, x = shape_org
    z_, y_s, x_s = shape_subres

    rf_x = int(x/x_s)
    rf_y = int(y/y_s)

    sections = dict()

    for section, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        sections[section] = {
            'x': [int(x*rf_x), int(x+(w*rf_x))],
            'y': [int(y*rf_y), int(y+(h*rf_y))]
        }
    
    return sections


def filter_by_location(
    table, #: Union[PDF, GDF],
    coords: dict,
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
        pixelsize : float
            Size of a pixel in X & Y. Optional.

    Returns
    -------
        out : polars or geopandas dataframe
            Same table as input, excluding all entries with coordinates not in `coords`.
    """
    assert depth(coords) == 1, \
        f'`coords` incorrect: {coords}\n Consult example .json!'
    if isinstance(table, pl.DataFrame):
        for axis, extremes in coords.items():
            var_min, var_max = extremes

            if pixelsize:
                var_min = var_min * pixelsize
                var_max = var_max * pixelsize
            if f'{axis}_location' in table.columns:
                column = f'{axis}_location'
            elif f'vertex_{axis}' in table.columns:
                column = f'vertex_{axis}'
            else:
                column = 'x'

            # if pixelsize:
            #     pixeled = table.select(
            #         (pl.col(column) / pixelsize).round(0),
            #     )
            #     table = table.update(pixeled)

            expr = ((pl.col(column)).is_between(
                var_min, var_max, closed='both'
            ))
            table = table.filter(expr)
    elif isinstance(table, gpd.GeoDataFrame):
        x_min, x_max = coords['x']
        y_min, y_max = coords['y']

        if pixelsize:
            x_min = x_min * pixelsize
            x_max = x_max * pixelsize
            y_min = y_min * pixelsize
            y_max = y_max * pixelsize

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


def boundaries_to_mask(table, shape):
    data = dict()
    masks = np.zeros(shape)

    for cell_id in set(table['cell_id']):
        x = table[table['cell_id'] == cell_id]['vertex_x']
        y = table[table['cell_id'] == cell_id]['vertex_y']
        if len(x) >= 4 or len(y) >= 4:

            geometry = Polygon(zip(x,y))
            data.update(dict(
                cell_id = geometry
            ))
            mask = np.zeros_like(masks, np.uint8)
            cv2.fillPoly(mask, np.int32([geometry]), (1,1,1))
            masks[mask > 0] = cell_id
    
    return masks, data


def transcripts_for_ovrlpy(machine, file_path):
        if machine == 'xenium':
            transcripts = ovrlpy.io.read_Xenium(
                filepath=file_path,
                min_qv=None,
                remove_features=ovrlpy.io.XENIUM_CTRLS,
                n_threads=cpu_count(),
            )
        else:
            raise NotImplementedError(
                f"Not Implemented to deal with `{machine}`."
            )

        return transcripts
