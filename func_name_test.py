import os
import timeit
from pathlib import Path

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "16"

from rich.traceback import install

install(show_locals=True)

import tifffile
from phenium import analysis, io
from phenium.main import Phenium

xenium_dir = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260827__133139__20260827_ITCAIM_lung-spleen_run4/output-XETG00410__0065619__non_infected_lung__20260827__133207"
pheno_img = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260902_ITCAIM_Phenium_Slide8/Scan1/20260902_ITCAIM_Phenium_Slide8_Scan1.er.qptiff"
output_dir = "/data/cephfs-1/scratch/projects/sawitzki-phenium"
config_path = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/phenium/config.toml"
prior_type = "CP"
channels = ("DAPI", "CD45")
exclude = [
    "CD69",
    "CD206",
    "TCRgd",
    "CD16",
    "Osteopontin",
    "FoxP3",
    "CD11c",
    "CD36",
    "CD62L",
    "TCRab",
]

ana = Phenium(
    SRT_dir=xenium_dir,
    IHC_path=pheno_img,
    output_dir=output_dir,
    config_path=config_path,
    prior_type=prior_type,
    channels=channels,
    exclude=exclude,
)

ana.prepare(4, 12, quick=True, verbose=True)

import math
from functools import partial
from itertools import product

import numpy as np
from shapely import Polygon

ana.cluster(resolution=0.2, save=True)

gdf = ana.assign_celltype(None)
celltype_prot_dict = dict(zip(np.random.randint(0,len(ana.adata.uns["leiden_colors"]),size=len(ana.IHC.roi_biomarkers),).astype(str),ana.IHC.roi_biomarkers))
biomarkers = ana.IHC.roi_biomarkers
roi = ana.IHC.roi.copy()
tile_size = 128

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
    new_img = np.zeros((c, 3, y, x))
    print('roi:', roi[channels, miny:maxy, minx:maxx].sum())
    print('roi/l:', (roi[channels, miny:maxy, minx:maxx]/layers).sum())
    new_img[channels, layer, miny:maxy, minx:maxx] += (roi[channels, miny:maxy, minx:maxx] / layers)
    print('new_img: ', new_img.sum())
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

nim = func_name(gdf, roi, tile_size, biomarkers, celltype_prot_dict)
print(nim.shape)
print(nim.sum())
print(nim.mean())
