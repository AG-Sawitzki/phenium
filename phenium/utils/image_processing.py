import os
from math import sqrt
from pathlib import Path
from itertools import product
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import cv2
import tifffile
import numpy as np

from typing import Any, Union
from pathlib import PosixPath
from numpy.typing import ArrayLike


def make_area_centres(
    n_roi: int,
    shape: tuple,
)-> list:
    """
    Fit `n_roi` squares equally distributed into `shape`.

    Relation between x & y decides squares per axis.
    Claculates the location of the centres of these squares.

    Parmeters
    ---------
        n_roi : int
            Number of expected punches/ROIs in an image.
        shape : tuple
            Shape of an Image with corresponding `n_roi`.

    Results
    -------
        out : list
            List of x,y coordinates. Centres of `n_roi` squares.

    """
    x,y = sorted(shape)
    print(x,y)
    partitions_x = int(sqrt(x/y*n_roi))
    partitions_y = int(n_roi/partitions_x)
    print(partitions_x, partitions_y)
    x_size = int(x/partitions_x)
    y_size = int(y/partitions_y)
    print(x_size, y_size)
    areas = list(product(
        range(0, y-y % y_size, y_size),
        range(0, x-x % x_size, x_size)
    ))
    centres = []
    for idx in range(len(areas)):
        coords_org = np.array(areas[idx])
        coords_ext = coords_org + np.array([x_size, y_size])
        print(coords_ext)
        centres.append(
            [coords_ext[1]-(coords_ext[1]-coords_org[1])//2,
             coords_ext[0]-(coords_ext[0]-coords_org[0])//2]
        )
    return centres


def get_area_index(
    centre: Union[list, tuple, ArrayLike],
    centres: Union[list, ArrayLike]
)-> int:
    """
    Retruns the Index of the nearest coordinates in a list.

    Parameters
    ----------
        centre : list, tuple or ArrayLike
            x,y coordinates.
        centres : list, ArrayLike
            List of x,y coordinates of centres of `n_roi` squares.

    Returns
    -------
        out : int
            Index of coordinates closest to `centre`.
            I.e. `n_roi` number.
    """
    relative_centres = np.abs(np.array(centres) - np.array(centre))
    tuple_relative_centres = [(coords[0], coords[1]) for coords in relative_centres]
    closest = min(tuple_relative_centres)
    idx = tuple_relative_centres.index(closest)
    return idx


def find_rois(
    shape_org: tuple,
    image_subres: ArrayLike,
    n_roi: int
) -> Union[list, ArrayLike]:
    """
    Sort the contours by area.

    Parameters:
    ----------
        shape_org : tuple
            Max resolution of img.
        image_subres : ArrayLike
            Lowest subresolution of image.
        n_roi : int
            Expected # of regions of interest.
            Should be equivalent to the number of tissue-samples on the slide.

    Returns
    ----------
        out : ArrayLike or list
            Contours of significant size.
    """
    z, y, x = shape_org

    subres_centre = np.uint8(image_subres[z//2])

    subres_dilated = cv2.dilate(
        subres_centre,
        np.ones((3, 3)),
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

    # keep contours with significant size
    values = []
    dtype = [('area', float), ('ai', float), ('y', float), ('x', float)]

    centres = make_area_centres(n_roi, subres_centre.shape)
    print(centres)
    for c in contours:
        (x, y), (w, h), a = cv2.minAreaRect(c)
        ai = get_area_index([x,y],centres)
        print(x,y, ai)
        values.append((w*h, ai, y, x))

    values_arr = np.array(values, dtype=dtype)
    # sort by area and find smallest allowed roi
    values_arr_sorted = np.sort(values_arr, kind='stable', order='area')
    smallest_allowed_roi = values_arr_sorted['area'][-n_roi]
    # sort by weighted_distance
    values_arr_ai_args = np.argsort(values_arr, kind='stable', order=['ai','y'])
    values_arr_ai_sorted = np.sort(values_arr, kind='stabe', order=['ai','y'])
    contours_ai_sorted = [contours[index] for index in values_arr_ai_args]

    mask = values_arr_ai_sorted['area'] >= smallest_allowed_roi
    nroi_contours = [
        contours_ai_sorted[index] for index, boolean in enumerate(mask)
        if boolean
    ]

    return nroi_contours, subres_centre