import itertools
import math
import os
import time
import xml.etree.ElementTree as ET
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path, PurePath
from typing import overload

import anndata as ad
import ants
import cv2
import geopandas as gpd
import numpy as np
import ovrlpy
import polars as pl
import polars.selectors as cs
import scanpy as sc
import squidpy as sq
import tifffile
from cellpose.utils import outlines_list
from numpy.typing import ArrayLike
from PIL import Image, ImageDraw
from shapely import Polygon
from skimage.filters import sobel
from skimage.transform import rescale

from phenium.exceptions import NotImplemented


def depth(
    d: dict
) -> int:
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0


def chunk_size(var: int, chunks: int) -> int:
    """Calculate the size of a chunk of a region.

    Parameters
    ----------
        var : int
            Total region width or height.
        chunks : int
            Total numbr of chunks.

    Returns
    -------
        out : int
            Width or Height of chunk.
    """
    return int(math.prod([var, math.sqrt(chunks) / chunks]))


def to_cli(
    tup: tuple[str, bool | int | float | str],
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

    img_subres = np.float64(img_subres[0, ...])
    img_subres /= img_subres.max()
    img_subres = np.uint8(img_subres * 255)
    img_subres = cv2.dilate(img_subres, np.ones((3, 3)), iterations=3)
    _, img_subres = cv2.threshold(
        img_subres,
        int(np.average(img_subres)),
        255, 0
    )
    contours, _ = cv2.findContours(
        img_subres,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = np.zeros((len(contours), 4))
    scores = np.zeros(len(contours))
    for idx, c in enumerate(contours):
        (x, y), (w, h), _a = cv2.minAreaRect(c)
        dim = max(w, h)
        bboxes[idx] = [x, y, dim, dim]
        scores[idx] = w*h
    scores /= max(scores)

    ids_to_keep = cv2.dnn.NMSBoxes(
        bboxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.1,
        nms_threshold=0.0,
    )

    bboxes_to_keep = bboxes[ids_to_keep]
    bboxes_sorted = sorted(bboxes_to_keep, key=lambda k: [int(k[1]/100), k[0]])

    return bboxes_sorted


@overload
def bboxes_to_sections(
    bboxes: list[ArrayLike],
    shape_org: tuple,
    subres: tuple,
    buffer: int = 128
) -> dict:
    pass
@overload
def bboxes_to_sections(
    bboxes: list[ArrayLike],
    shape_org: tuple,
    subres: ArrayLike,
    buffer: int = 128
) -> tuple[dict, ArrayLike]:
    pass
def bboxes_to_sections(
    bboxes,
    shape_org,
    subres,
    buffer = 128
):
    """
    Takes bounding boxes from cv2 and organizes the coordinates into a dictionary.
    """
    _C, Y, X = shape_org
    if isinstance(subres, tuple):
        _C_s, Y_s, X_s = subres
    else:
        marked_img = np.uint8(subres[0, ...])
        Y_s, X_s = marked_img.shape

    rf_x = int(X/X_s)
    rf_y = int(Y/Y_s)

    sections = {
        idx: {'x': [0, 0], 'y': [0, 0]} for idx in range(len(bboxes))
    }

    for section, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        x_low = int(max(0, ((x - w / 2) * rf_x) - buffer))
        x_high = int(min(X, ((x + w / 2) * rf_x) + buffer))
        y_low = int(max(0, ((y - h / 2) * rf_y) - buffer))
        y_high = int(min(Y, ((y + h / 2) * rf_y) + buffer))
        sections[section] = {
            'x': [x_low, x_high],
            'y': [y_low, y_high]
        }
        if not isinstance(subres, tuple):
            cv2.rectangle(
                marked_img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)),
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


def scale_coords(coords, factor):
    """
    Scale the min/max values in `coords` by `factor`.

    Parameters
    ----------
        coords : dict
            Dictionary with keys 'x' & 'y' and values [min, max]
        factor : float or tuple[float, ...]
            Either a single factor for both or
            each axis receives a separate factor.
            Only the first two entries are used.
            Order: x_factor, y_factor.
    
    Returns
    -------
        out : dict
            A new dictionary with the old keys but values scaled by factor.
    """
    x = np.array(coords['x'])
    y = np.array(coords['y'])
    new_coords = {'x': [0, 0], 'y': [0, 0]}

    if isinstance(factor, tuple):
        if len(factor) == 2:
            x_factor, y_factor = factor
        elif len(factor) >= 3:
            x_factor, y_factor, _ = factor
        else:
            x_factor, y_factor = factor, factor
    elif isinstance(factor, float):
        x_factor, y_factor = factor, factor
    else:
        raise NotImplementedError('Unknown format of factor.')

    new_coords['x'] = np.int32(x * x_factor).tolist()
    new_coords['y'] = np.int32(y * y_factor).tolist()

    return new_coords


def relative(
    table: pl.DataFrame | gpd.GeoDataFrame,
    coords: dict,
) -> pl.DataFrame | gpd.GeoDataFrame:
    """
    Turn absolute coordinates into ones relative to the section origin.

    Parameters
    ----------
        table : polars or geopandas dataframe
            Contains the coordinates to adjust.
            Column with `vertex_` or `_location` for polars.
            `geometry` column in GeoDataFrame.
        coords : dict
            Dictionary containing min. x,y and max. x,y.
            For an example, see `gt_section.json`

    Returns
    -------
        out : polars or geopandas dataframe
            Same table as input, but with coordinates relative to section origin.
    """
    assert depth(coords) == 1, f"`coords` incorrect: {coords}\n Consult example .json!"
    if isinstance(table, pl.DataFrame):
        for axis, coord in coords.items():
            column = next(
                c
                for c in table.columns
                if (f"{axis}_location" in c or f"vertex_{axis}" in c)
            )
            var_min = min(coord)
            relatived = table.select((pl.col(column)) - var_min)
            table = table.update(relatived)
    if isinstance(table, gpd.GeoDataFrame):
        x_min = min(coords['x'])
        y_min = min(coords['y'])
        print('IN RELATIVE', table)
        geometry = table['geometry'].transform(
            lambda x: (x - np.array([x_min, y_min])).round(0)
        )
        table['geometry'] = geometry
    return table


def pixelate(
    table: pl.DataFrame | gpd.GeoDataFrame,
    pixelsize_xy: float,
) -> pl.DataFrame | gpd.GeoDataFrame:
    """
    Transform coordinates in (geo)dataframe from length unit to pixelsize.

    Assumes pixel have same length in x & y.

    Parameters
    ----------
        table : polars or geopandas dataframe
            Contains the coordinates to transform.
            Column with `vertex_` or `_location` for polars.
            `geometry` column in GeoDataFrame.
        pixelsize_xy : float
            Pixelsize of a pixel in xy plane.

    Returns
    -------
        out : polars or geopandas dataframe
            Same table as input, but with adjusted coordinates.
    """
    if isinstance(table, pl.DataFrame):
        columns = (
            (cs.ends_with('_location') & ~cs.starts_with('z'))
            | (cs.starts_with('vertex_') & ~cs.ends_with('z'))
        )
        pixeled = table.select(columns) / pixelsize_xy
        pixeled = pixeled.round(0)
        table = table.update(pixeled)
    if isinstance(table, gpd.GeoDataFrame):
        geometry = table['geometry'].transform(
            lambda x: (x * np.array([1 / pixelsize_xy, 1 / pixelsize_xy])).round(0)
        )
        table['geometry'] = geometry
    return table


def filter_by_location(
    table: pl.DataFrame | gpd.GeoDataFrame,
    coords: dict,
    z_bounds: tuple | None = None,
    pixelsize: float | None = None,
) -> pl.DataFrame | gpd.GeoDataFrame:
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
                print(f'Setting col to {axis}')
                column = axis

            expr = ((pl.col(column)).is_between(
                var_min, var_max, closed='both'
            ))
            table = table.filter(expr)
            # print(axis, extremes)
            # print(table.shape)
        if z_bounds != None and 'z_location' in table.columns:
            z_low, z_high = z_bounds
            expr = ((pl.col('z_location')).is_between(
                z_low, z_high, closed='both'
            ))
            table = table.filter(expr)
            # print(z_low, z_high)
            # print(table.shape)
    elif isinstance(table, gpd.GeoDataFrame):
        x_min, x_max = coords['x']
        y_min, y_max = coords['y']

        if pixelsize:
            x_min *= pixelsize
            x_max *= pixelsize
            y_min *= pixelsize
            y_max *= pixelsize

        polygon = Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_min, y_max), (x_max, y_max),
            (x_min, y_min),
        ])
        check = table['geometry'].within(polygon)
        table = table[check]
        if table.index.name in table.columns:
            table.drop(table.index.name, axis=1, inplace=True)
    else:
        raise NotImplementedError(
            f'Not Implemented to deal with {type(table)}.'
        )

    return table


def denoise_image(image, noise_model="Rician"):
    image = np.int32(image)

    image_ant = ants.from_numpy(image)
    image_denoised = ants.ops.denoise_image(
        image_ant,
        mask=None,
        shrink_factor=1,
        p=1,
        r=2,
        noise_model=noise_model,
    )

    return image_denoised.numpy()


def transform_image(
    moving,
    fixed,
    scaling_factor,
    reg_fwd
):
    moving = rescale(
        moving,
        scaling_factor,
        preserve_range=True,
    )
    if isinstance(fixed, tuple):
        fixed = np.zeros(fixed, dtype=np.int32)
    else:
        fixed = np.int32(fixed)
    moving = np.int32(moving)

    fixed_ant = ants.from_numpy(fixed)
    moving_ant = ants.from_numpy(moving)

    moving_transformed = ants.apply_transforms(
        fixed=fixed_ant,
        moving=moving_ant,
        transformlist=reg_fwd,
        interpolator="nearestNeighbor",
    )

    # moving_denoised = ants.ops.denoise_image(
    #     moving_transformed,
    #     mask=None,
    #     shrink_factor=1,
    #     p=1,
    #     r=2,
    #     noise_model='Rician'
    # )

    return moving_transformed.numpy()


def value_at_cell(
    cell_id,
    masks,
    roi
):
    if cell_id != 0:
        fltr = masks == cell_id
        vals = roi[:, fltr].sum(axis=1)
        return vals


def cell_loc(
    cell_id,
    masks,
):
    if cell_id != 0:
        _, x = masks.shape
        fltr = masks == cell_id
        idx = np.argmax(fltr)
        return np.array([math.fmod(idx, x), math.floor(idx / x)])


# modified `get_outline_multi` from cellpose.utils
def cell_contour(
    cell_id,
    masks,
):
    """
    Get the outline of a specific mask in a multi-mask image.

    Parameters
    ----------
        cell_id : int
            identification of the mask to find the outline of
        masks : np.ndarray
            Array containing at least the entire mask of `cell_id`.

    Returns
    -------
        out : shapely.Polygon
            A shapely polygon of the outline (x, y).
    """
    if cell_id != 0:
        bmask = masks == cell_id
        if bmask.sum() > 3:
            cntrs = cv2.findContours(
                bmask.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE,
            )
            cntrs = cntrs[-2]
            cmax = np.argmax([c.shape[0] for c in cntrs])
            pix = cntrs[cmax].astype(int).squeeze()
            if len(pix) > 4:
                return Polygon(zip(pix[:, 1], pix[:, 0]))


def _draw_mask(
    idx,
    table,
    cell_ids,
    masks,
    pxsize,
):
    cell_id = cell_ids[idx]
    table_cid = table.filter(pl.col('cell_id') == cell_id)
    x = table_cid.select(pl.col('vertex_x')).to_numpy().ravel()
    y = table_cid.select(pl.col('vertex_y')).to_numpy().ravel()
    px_x = np.int32(x/pxsize[0])
    px_y = np.int32(y/pxsize[1])
    polyline = list(zip(px_x, px_y))
    img = Image.new('1', masks.shape, 0)
    ImageDraw.Draw(img).polygon(polyline, outline=1, fill=1, width=1)
    masks[np.array(img)] = idx

    return masks


def boundaries_to_mask(
    table: pl.DataFrame,
    shape: tuple,
    pxsize: tuple,
) -> ArrayLike | dict:
    """
    ...

    Parameters
    ----------
        table : polars DataFrame
            polars dataframe with `cell_id` and `vertex_(x,y)` columns
        shape : tuple
            shape of the resulting image
        pxsize : tuple
            tuple of size of a pixel in x, y direction
    """
    masks = np.zeros(shape[1:], dtype=np.uint64)

    unique_ids = table.select(pl.col('cell_id')).unique()
    cell_ids = [row['cell_id'] for row in unique_ids.iter_rows(named=True)]
    cell_ids = sorted(cell_ids)

    # for idx in tqdm.tqdm(range(len(cell_ids))):
    #     masks = _draw_mask(
    #         idx,
    #         table,
    #         cell_ids,
    #         masks,
    #         pxsize,
    #     )

    func = partial(
        _draw_mask,
        table=table,
        cell_ids=cell_ids,
        masks=masks,
        pxsize=pxsize,
    )
    iterations = range(len(cell_ids))

    with Pool(processes=2) as pool:
        res = pool.imap_unordered(
            func,
            iterations,
            chunksize=3,
        )
        pool.close()
        pool.join()
    print(res)
    print(sum(sum(masks)))
    return masks


def transcripts_for_ovrlpy(
    machine: str,
    transcripts: pl.DataFrame | os.PathLike | PurePath,
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
    img_path: str | os.PathLike | PurePath
) -> tuple[float, float, float]:
    """
    Get pixel size from xml metadata of (OME)Tiff.
    Returns `-1` for nonexisting Physical Sizes.

    Parameters
    ----------
        img_path : str, os.PathLike, PurePath
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
        px_size_Y = pixels.get('PhysicalSizeY', default=px_size_X)
        px_size_Z = pixels.get('PhysicalSizeZ', default=-1)
    elif pixels is None:
        px_size_X = tags[282].value[1]/tags[282].value[0]*1e4
        px_size_Y = tags[283].value[1]/tags[283].value[0]*1e4
        px_size_Z = -1
    else:
        raise ValueError('Cannot find the pixelsize.')

    return float(px_size_X), float(px_size_Y), float(px_size_Z)


def get_biomarkers(img_path):
    # function written based on grenkoca/qptifffile.
    biomarkers = {}
    suffix = Path(img_path).suffix
    if suffix == '.tif':
        with tifffile.TiffFile(img_path) as tiff:
            tags = tiff.pages[0].tags
    
        root = ET.fromstring(tags[270].value)
        if root.findall('.//*[@SizeC]'):
            channels = root.findall('.//*[@Name]')
            for channel in channels:
                name = channel.get('Name')
                idx = channel.get('ID').replace('Channel:', '')
                biomarkers[name] = int(idx)
    elif suffix == '.qptiff':
        with tifffile.TiffFile(img_path) as tif:
            pages = tif.series[0].pages

        biomarker_paths = [
            './/Biomarker',
            './/BioMarker',
            './/BioMarker/Name',
            './/Biomarker/Name',
            './/StainName',
            './/Marker',
            './/ProteinMarker'
        ]

        for page_idx, page in enumerate(pages):
            if hasattr(page, 'description') and page.description:
                root = ET.fromstring(page.description)
                for path in biomarker_paths:
                    idx = biomarker_paths.index(path)
                    element = root.find(path)
                    if element is not None and element.text:
                        biomarkers[element.text] = page_idx
                        if idx > 0:
                            biomarker_paths.insert(
                                0,
                                biomarker_paths.pop(idx)
                            )
                        break

        if len(biomarkers) != len(pages):
            print('Not all channels have a biomarker.')

    return biomarkers

def run_leiden(
    adata,
    resolution: float = 0.25,
    percent_top=(25, 75, 150),
    n_comps=100,
    n_pcs=40
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
        percent_top=percent_top,
        inplace=True,
        log1p=True
    )

    sc.pp.filter_cells(adata, min_genes=3)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.tl.pca(adata, n_comps=n_comps)

    # if gpu:
    #     rsc.get.anndata_to_GPU(adata, convert_all=True)
    #     rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    #     rsc.tl.umap(adata)
    #     rsc.tl.leiden(adata, resolution=resolution)
    #     rsc.get.anndata_to_CPU(adata, convert_all=True)
    # else:
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor='igraph', n_iterations=2)

    return adata


# function form stackoverflow
# adapted to return shapely Polygons
def process_roi(masks: ArrayLike) -> gpd.GeoDataFrame:
    """
    Get the polgyons from the prediction-masks using cellpose.utils functions.

    Saves them as a GeoDataFrame (geojson)

    Parameters
    ----------
        npy_data : ArrayLike
            The numpy.ndarray of the masks.
        npy_base_output_path : Path
            Path to save the geojson.

    Returns
    ----------
        out : None
            Automatically saves the GDF under npy_base_output_path.
    """
    print(" - Extracting ROI")
    masks = masks.squeeze()
    # change the index order:
    # first the cell then the layer it is on.
    # thus one would know how the same cell looks on different layers
    data = {"cell": [], "layer": [], "geometry": []}
    if masks.ndim == 3:
        zs = min(masks.shape)
        posZ = masks.shape.index(zs)
        for z in range(zs):
            if posZ == 0:
                mask_z = masks[z, ...]
            elif posZ == 1:
                mask_z = masks[:, z, :]
            else:
                mask_z = masks[..., z]
            print(f" - Layer {z}")
            coords_list = outlines_list(mask_z)
            i = 1
            for coords in coords_list:
                data["layer"].append(z)
                data["name"].append(f"cell_{i}")
                data["geometry"].append(Polygon(coords))
                i += 1
    else:
        coords_list = outlines_list(masks)
        i = 1
        for coords in coords_list:
            data["layer"].append(np.nan)
            data["cell"].append(f"cell_{i}")
            data["geometry"].append(Polygon(coords))
            i += 1
    gdf = gpd.GeoDataFrame(data=data)
    gdf.set_index(["layer", "name"])
    print(" - Saving GeoDataFrame")
    return gdf


# from https://medium.com/@amine-RT/guide-to-image-noise-reduction-techniques-in-python-c24da16052dc
def adaptive_median_filter(pixels):
    center = pixels[len(pixels) // 2]
    sorted_pixels = sorted(pixels)
    median = sorted_pixels[len(sorted_pixels) // 2]
    return median if center == 0 or center == 255 else center


def foo(bnds, roi, gdf, biomarkers, celltype_prot_dict):
    minx, miny, maxx, maxy, idx = bnds
    cell, layer, _geom, ct = gdf.loc[idx]
    layers = gdf[gdf.cell == cell].layer.size
    prots = celltype_prot_dict[ct]
    if isinstance(prots, list):
        channels = [
            int(biomarkers[c]) for c in prots
        ]
    elif isinstance(prots, str):
        channels = [
            int(biomarkers[prots])
        ]
    else:
        raise TypeError('content of celltype_prot_dict is wrong type: ', type(prots))
    c, y, x = roi.shape
    z = gdf.layer.max
    new_img = np.zeros((c, 3, y, x), dtype=roi.dtype)
    new_img[channels, layer, miny:maxy, minx:maxx] += (roi[channels, miny:maxy, minx:maxx] / layers)
    return new_img


def func_name(gdf, roi, tile_size, biomarkers, celltype_prot_dict):
    if roi.ndim == 3:
        _c, y, x = roi.shape
    elif roi.ndim == 4:
        _z, _c, y, x = roi.shape
    else:
        raise NotImplementedError("What is this shape?")
    if isinstance(tile_size, tuple):
        x_size, y_size = tile_size
    else:
        x_size = y_size = tile_size
    x_range = range(0, int(x - math.fmod(x, x_size)), x_size)
    y_range = range(0, int(y - math.fmod(y, y_size)), y_size)
    grid = product(y_range, x_range)
    new_img = np.zeros(
        (
            len(biomarkers),
            3,
        )
        + roi.shape[~1:]
    )
    for (y_c, x_c) in grid:
        y_low, y_high = y_c, y_c + y_size
        x_low, x_high = x_c, x_c + x_size
        plgn = Polygon([
            (x_low, y_low), (x_high, y_low),
            (x_low, y_high), (x_high, y_high),
            (x_low, y_low)
        ])
        sub = gdf[gdf.geometry.intersects(plgn)]
        if sub.size > 0:
            bnds = sub.bounds.astype(int)
            bnds['i'] = sub.index
            bnds = bnds.to_numpy()
            func = partial(
                foo,
                roi=roi[:, y_low:y_high, x_low:x_high],
                gdf=gdf,
                biomarkers=biomarkers,
                celltype_prot_dict=celltype_prot_dict,
            )
            res = np.apply_along_axis(func, 1, bnds)
            new_img[:, :, y_low:y_high, x_low:x_high] = res.sum(axis=0)
    return new_img
