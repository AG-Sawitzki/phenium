import gc
import math
import random

# from copy import deepcopy
from functools import partial
from itertools import (
    # chain,
    product,
)
from multiprocessing import (
    Pool,
    # cpu_count
)
from pathlib import Path, PurePath

import anndata as ad
import ants
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import tifffile
import tomlkit
import torch
import tqdm
import zarr
from cellpose import models
from matplotlib.axes import Axes
from scipy.io import loadmat, mmread
from scipy.sparse import csr_matrix
from shapely import transform
from skimage.draw import polygon
from skimage.transform import rescale

from phenium.analysis import ovrlpy_analysis, proseg_doublets, vis_doublets
from phenium.io import load_plgns, save_anndata, save_settings
from phenium.plot import polygon_overlay, scatter_doublets
from phenium.segment import proseg_wrapper
from phenium.utils import (
    adjust_z,
    apply_factor,
    bboxes_to_sections,
    boundaries_to_mask,
    cell_loc,
    chunk_size,
    filter_by_location,
    find_rois,
    get_biomarkers,
    get_pixel_size,
    pixelate,
    process_roi,
    relative,
    run_leiden,
    scale_coords,
    transform_image,
    value_at_cell,
)


class IMG:
    """
    multi-channel img
    (c, y, x) or (c, z, y, x)
    """
    __slots__ = [
        "_roi",
        "_roi_adj",
        "adata",
        "affine_transform",
        "cell_loc",
        "feature_matrix",
        "masks",
        "overview",
        "path",
        "plgns",
        "reg_factor",
        "reg_fwd",
        "reg_inv",
        "reg_shape",
        "sections",
        "selection",
        "zrr"
    ]
    def __init__(
        self,
        path,
        exclude: list[(str | int), ...] | None = None
    ):
        self.path = Path(path)

        if not isinstance(exclude, list):
            exclude = [exclude]
        
        self.selection = [
            c for k, c in self.all_biomarkers.items() if (
                (c not in exclude) & (k not in exclude)
            )
        ]

        store = tifffile.imread(self.path, aszarr=True)
        self.zrr = zarr.open(store, mode="r")

        bboxes = find_rois(
            self.shape,
            self.zrr[str(self.lvl-2)]
        )
        self.sections, self.overview = bboxes_to_sections(
            bboxes,
            self.shape,
            self.zrr[str(self.lvl-2)]
        )

    @property
    def lvl(self):
        """
        The count of resolutions in the zarr-object. 
        """
        return len(self.zrr)

    @property
    def shape(self):
        """
        Shape of the highest resolution in the zarr-object
        """
        return self.zrr["0"].shape

    @property
    def axes(self):
        """
        At what position in the is a axis located.
        """
        with tifffile.TiffFile(self.path) as tif:
            axes = tif.series[0].axes
        axs = {}
        for ax in ['X', 'Y', 'Z', 'C', 'Z', 'T']:
            pos = axes.index(ax) if ax in axes else None
            if pos is not None:
                axs[ax] = pos
        return axs

    @property
    def pxsize(self):
        """
        Size of a pixel in x,y and z axis.
        """
        return get_pixel_size(self.path)

    @property
    def roi(self):
        """
        Numpy array of sub-section in highest resolution zarr-object.
        """
        try:
            return self._roi_adj
        except AttributeError:
            return self._roi

    @roi.setter
    def roi(self, section):
        coords = self.sections[section]
        x = coords["x"]
        y = coords["y"]
        self._roi = self.zrr["0"][self.selection, y[0]:y[1], x[0]:x[1]]

    @property
    def all_biomarkers(self):
        """
        All biomarkers of the image. Before 'exclude'.
        """
        biomarkers = get_biomarkers(self.path)
        return biomarkers

    @property
    def roi_biomarkers(self):
        """
        Biomarkers of the region of interest. After 'exclude'.
        """
        biomarkers = {
            k: v for k, v in self.all_biomarkers.items() if v in self.selection
        }
        biomarkers = {
            k: v for v, k in enumerate(biomarkers.keys())
        }
        return biomarkers


    def find_affine(
        self,
        fixed: np.ndarray,
        scaling_factor: float,
        quick: bool = False,
        matrix: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Find the matrix to affine-transform this image to register with `fixed`.

        Parameters
        ----------
            fixed: numpy.ndarray
                Array of image to register `self` to.
            scaling_factor: float
                Factor to scale `self` to be of the same size as `fixed`.
            quick: bool
                If `True` uses 'antsRegistrationSyNQuick[r]' and not 'antsRegistrationSyN[r]'.
                Default is `False`.
            matrix: bool
                If `True` assigns the variable `affine_transform`.
                Default is `False`.
            verbose: bool
                If `True` prints the ANTsPy verbose-output.
                Default is `False`.

        Returns
        -------
            out: None
                Assigns the forward and reverse transform-matrix, the scaling factor and new shape.
        """
        moving = rescale(
            self.roi[0, :, :],
            scaling_factor,
            preserve_range=True,
        )
        moving = np.flip(moving)

        fixed = np.uint32(fixed)
        moving = np.uint32(moving)

        fixed_ant = ants.from_numpy(fixed)
        moving_ant = ants.from_numpy(moving)

        if quick:
            tof = "antsRegistrationSyNQuick[r]"
        else:
            tof = "antsRegistrationSyN[r]"

        reg = ants.registration(
            fixed=fixed_ant,
            moving=moving_ant,
            type_of_transform=tof,
            verbose=verbose,
        )

        if matrix:
            mat = loadmat(reg["fwdtransforms"][0])
            keys = list(mat.keys())

            dim = fixed.ndim
            A = mat[keys[0]][: dim**2].reshape((dim, dim))
            t = mat[keys[0]][dim**2 :]
            c = mat[keys[1]]

            w = t + c - np.dot(A, c)
            Ah = np.identity(dim + 1)
            Ah[:dim, :dim] = A
            Ah[:dim, dim:] = w
            self.affine_transform = Ah

        self.reg_fwd = reg["fwdtransforms"]
        self.reg_inv = reg["invtransforms"]
        self.reg_factor = scaling_factor
        self.reg_shape = fixed.shape
        gc.collect()


    def adjust(
        self,
        parallel: bool = True,
        return_copy: bool = False,
        verbose: bool = False,
    ) -> None | np.ndarray:
        """
        Adjusts the roi to fit the input image from `find_affine`.

        Pramaters
        ---------.
            parallel: bool
                If `True` splits the process into two, running parallel.
                Default is `True`.
            return_copy: bool
                If `True` returns the adjusted image and does not update `roi`.
                Default is `False`.
            verbose: bool
                If `True` prints a progress-bar indicating time and layer.
                Default is `False`

        Returns
        -------
            out: None or numpy.ndarray
                If `return_copy` the adjusted image is returned. Otherwise it replaces `roi`.
        """
        _ = torch.cuda.empty_cache()
        iterations = list(self.roi_biomarkers.values())
        moving = self.roi.copy() if parallel else self.roi
        # self._roi_adj = np.zeros((len(iterations),)+fixed.shape)
        imgs = []
        func = partial(
            transform_image,
            fixed=self.reg_shape,
            reg_fwd=self.reg_fwd,
            scaling_factor=self.reg_factor,
        )
        if parallel:
            if verbose:
                res = []
                with Pool(processes=2) as pool:
                    for r in tqdm.tqdm(
                        pool.imap(func, moving),
                        total=len(iterations),
                    ):
                        res.append(r)  # noqa: PERF402
            else:
                with Pool(processes=2) as pool:
                    res = pool.imap(func, moving)
                    pool.close()
                    pool.join()
            imgs = res
        else:
            if verbose:
                pbar = tqdm.tqdm(total=len(iterations))
                pbar.set_description('Adjusting Channel: X')
            for channel in iterations:
                if verbose:
                    pbar.set_description(f"Adjusting Channel: {channel}")
                img = func(moving[channel])
                imgs.append(img)
                if verbose:
                    pbar.update(1)
            if verbose:
                pbar.close()

        imgs_stacked = np.dstack(imgs)
        imgs_stacked_moved = np.moveaxis(imgs_stacked, -1, 0)
        gc.collect()
        if return_copy:
            return imgs_stacked_moved
        else:
            self._roi_adj = imgs_stacked_moved

    def segment(
        self,
        channels: list[int | str, ...] | tuple[int | str, ...],
        chunks: int = 4,
        return_copy: bool = False,
        verbose: bool = False,
        # mkwargs,
        # ekwargs
        gpu=True,
        pretrained_model="cpdino",
        device="cuda:0",
        use_bfloat16=True,
        batch_size=8,
        resample=True,
        channel_axis=None,
        z_axis=None,
        cellprob_threshold=0.0,
        do_3D=False,
        flow3D_smooth=0,
        stitch_threshold=0.0,
        min_size=15,
        max_size_fraction=0.4,
        augment=False,
        tile_overlap=0.1,
        compute_masks=True,
    ) -> None | np.ndarray:
        """
        Segment the region of interest using Cellpose.

        Parameters
        ----------
            channels: list[int | str, ...] or tuple[int | str, ...]
                Channels to use as input for segmentation.
            chunks: int
                Amount of sub-images the roi is split into.
                Must be a square number.
                Default is 4.
            return_copy: bool
                If `True` returns the mask of roi and does not add `masks`.
                Default is `False`.
            verbose: bool
                If `True` prints a progress-bar indictating time and sub-image.
                Default is `False`.

        Returns
        -------
            out: None or numpy.ndarray
                If `return_copy` then an array containing the masks is returned. Otherwise `masks` is added.

        Cellpose-Arguments
        ------------------
            See https://github.com/MouseLand/cellpose/cellpose/models.py for a description of parameters.
        """
        if not isinstance(channels, list):
            channels = list(channels)

        if isinstance(channels[0], int):
            list(channels)
        elif isinstance(channels[0], str):
            channels = [
                self.roi_biomarkers[c] for c in channels
            ]
        else:
            raise TypeError(
                "`channels` has wrong type: \n"
                f"\t {type(channels[0])}"
            )

        roi = self.roi[channels, :, :].copy()

        if self.roi.ndim == 3:
            _c, y, x = self.roi.shape
            channel_axis = 0
            z_axis = None
        elif self.roi.ndim == 4:
            _z, _c, y, x = self.roi.shape
            channel_axis = 1
            z_axis = 0
        else:
            raise NotImplementedError("What is this shape?")

        x_size = chunk_size(x, chunks)
        y_size = chunk_size(y, chunks)

        x_range = range(0, int(x - math.fmod(x, x_size)), x_size)
        y_range = range(0, int(y - math.fmod(y, y_size)), y_size)

        grid = product(y_range, x_range)

        vrlpy = max(0.05*y, 32)
        vrlpx = max(0.05*x, 32)
        all_masks = np.zeros((y, x), dtype=np.uint16)

        _ = torch.cuda.empty_cache()

        model = models.CellposeModel(
            gpu=gpu,
            pretrained_model=pretrained_model,
            device=torch.device(device),
            use_bfloat16=use_bfloat16,
        )
        if verbose:
            pbar = tqdm.tqdm(total=chunks)
        for chunk, (y_c, x_c) in enumerate(grid):
            if verbose:
                pbar.set_description(f'Segmenting {chunk}/{chunks}')
            y_low = max(0, int(y_c - vrlpy))
            y_high = min(y, int(y_c + y_size + vrlpy))

            x_low = max(0, int(x_c - vrlpx))
            x_high = min(x, int(x_c + x_size + vrlpx))

            roi_chunk = roi[
                :, y_low:y_high, x_low:x_high
            ]
            masks, _flows, _styles = model.eval(
                roi_chunk,
                resample=resample,
                channel_axis=channel_axis,
                z_axis=z_axis,
                cellprob_threshold=cellprob_threshold,
                do_3D=do_3D,
                flow3D_smooth=flow3D_smooth,
                stitch_threshold=stitch_threshold,
                min_size=min_size,
                max_size_fraction=max_size_fraction,
                augment=augment,
                tile_overlap=tile_overlap,
                compute_masks=compute_masks,
            )
            masks[masks != 0] += all_masks.max()
            all_masks[y_low:y_high, x_low:x_high] = masks
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

        del model
        self.masks = all_masks
        gc.collect()
        if return_copy:
            return all_masks

    def make_feature_matrix(
        self,
        masks: np.ndarray | None = None,
        tile_size: tuple[int, int] | int = 64,
        verbose: bool = False,
    )-> None:
        """
        Compute the feature-matrix of roi using `masks`.

        Parameters
        ----------
            masks: numpy.ndarray or None
                The segmentation-mask to use for constructing the feature-matrix.
                If not given, tries using self.masks.
                Optional.
            tile_size: tuple[int, int] or int
                Size of a tile in (x,y) or x=y direction.
                Default is 64.
            verbose:
                If `True` prints a progress-bar indicating time and tile.
                Default is `False`.
        
        Returns
        -------
            out: None
                Assigns the matrix to `feature_matrix` and the cell-locations to `cell_loc`.
        """
        if masks is None:
            masks = self.masks.copy()
        if masks.shape != self.roi.shape[1:]:
            self.adjust()

        masks = masks.astype(int)

        if self.roi.ndim == 3:
            _c, y, x = self.roi.shape
        elif self.roi.ndim == 4:
            _z, _c, y, x = self.roi.shape
        else:
            raise NotImplementedError("What is this shape?")

        if isinstance(tile_size, tuple):
            x_size, y_size = tile_size
        else:
            x_size = y_size = tile_size

        
        x_range = range(0, int(x - math.fmod(x, x_size)), x_size)
        y_range = range(0, int(y - math.fmod(y, y_size)), y_size)
        grid = product(y_range, x_range)
        dense_fmatrix = np.zeros((np.max(masks)+1, len(self.roi_biomarkers)))
        cell_locs = np.zeros((np.max(masks) + 1, 2))

        if verbose:
            pbar = tqdm.tqdm(total=(len(x_range) * len(y_range)))
        for idx, (y_c, x_c) in enumerate(grid):
            if verbose:
                pbar.set_description('Assigning features to cell in {idx}')
            y_low = y_c
            y_high = y_c + y_size
            x_low = x_c
            x_high = x_c + x_size

            mask = masks[y_low:y_high, x_low:x_high]

            if np.count_nonzero(mask) > 0:
                roi = self.roi[:, y_low:y_high, x_low:x_high]
                func_fm = partial(value_at_cell, masks=mask, roi=roi)
                func_loc = partial(cell_loc, masks=mask)

                cell_ids = np.trim_zeros(np.unique(mask, sorted=True))
                cell_ids_e = np.expand_dims(cell_ids, 1)

                res_fm = np.apply_along_axis(func_fm, 1, cell_ids_e)
                res_loc = np.apply_along_axis(func_loc, 1, cell_ids_e)

                dense_fmatrix[cell_ids] += res_fm
                cell_locs[cell_ids] = res_loc + [x_low, y_low]

                if verbose:
                    pbar.update(1)

        if verbose:
            pbar.close(1)
        self.feature_matrix = csr_matrix(dense_fmatrix)
        self.cell_loc = cell_locs.astype(np.int32)
        gc.collect()

    def find_cell_contour(
        self,
        masks: np.ndarray | None = None,
        tile_size: int = 64,
        verbose: bool = False,
    ):
        """
        Finds the contours of `masks` and assigns them in a geopandas.GeoDataFrame to `plgns`.

        Parameters
        ----------
            masks: numpy.ndarray or None
                The segmentation-mask to find the boundaries of.
                If not given, tries using self.masks.
                Optional.
            tile_size: tuple[int, int] or int
                Size of a tile in (x,y) or x=y direction.
                Default is 64.
            verbose:
                If `True` prints a progress-bar indicating time and cell_id.
                Default is `False`.
        
        Returns
        -------
            out: None
                Assigns the boundaries to `plgns`..
        """
        if masks is None:
            masks = self.masks.copy()

        if masks.ndim == 2:
            y, x = masks.shape
        else:
            _, y, x = masks.shape

        data = []
        if verbose:
            pbar = tqdm.tqdm(total=len(self.cell_loc))
        for cid, (x_c, y_c) in enumerate(self.cell_loc):
            if verbose:
                pbar.set_description(f"Finding contour for Cell {cid}")
            if cid != 0:
                y_low = max(0, y_c - tile_size)
                y_high = min(y, y_c + tile_size)
                x_low = max(0, x_c - tile_size)
                x_high = min(x, x_c + tile_size)

                mask = masks[y_low:y_high, x_low:x_high]
                mask = np.uint32(mask == cid)
                gdf = process_roi(mask, cell_id_start=cid)
                gdf.geometry = transform(gdf.geometry, lambda x: x + [x_low, y_low])  # noqa: B023
                data.append(gdf)
                if verbose:
                    pbar.update(1)
        if verbose:
            pbar.close()
        self.plgns = pd.concat(data)
        gc.collect()


    def _anndata(self):
        """
        Making an anndata obj.
        """
        counts = self.feature_matrix

        rows, _ = counts.shape
        obs = np.arange(rows).tolist()
        var = list(self.roi_biomarkers.keys())

        obs = pd.DataFrame({'cell':obs})
        var = pd.DataFrame({'protein':var})

        X = counts.tocsr()

        obsm = {"spatial": self.cell_loc}

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
            obsm=obsm
        )
        adata.var_names = adata.var.protein
        adata.obs_names = adata.obs.cell

        gc.collect()

        return adata

    def cluster(
        self,
        resolution: float = 0.175,
        exclude: list[str, ...] | tuple[str, ...] | None = None,
        masks: np.ndarray | None = None,
        return_copy: bool = False,
    )-> None | ad.AnnData:
        """
        Cluster the feature_matrix using Leiden.

        Parameters
        ----------
            resolution: float
                Resolution of the Leiden clustering.
                Default is 0.175
            exclude: list[str, ...], tuple[str, ...] or None
                Iterable of proteins to exclude from the clustering.
            masks: numpy.ndarray or None
                The masks to use to construct the feature-matrix from.
                If not given, tries `masks`.
                Optional.
            return_copy: bool
                If `True` returns the clustered anndata-object and does not add `adata`.
                Default is `False`.

        Returns
        -------
            out: None or anndata.AnnData
                If `return_copy` then the anndata-object is returned. Otherwise assigns the result to `adata`.
        """
        if masks is not None:
            masks = masks.astype(int)
        else:
            try:
                masks = self.masks.copy()
            except AttributeError:
                raise AttributeError('No mask given. IMG has no segmentation mask.')

        self.make_feature_matrix(masks=masks)

        adata = self._anndata()

        if exclude is not None and isinstance(exclude, (tuple, list)):
            adata = adata[:, ~adata.var_names.isin(exclude)].copy()

        adata = run_leiden(
            adata,
            resolution=resolution,
            percent_top=(3, 8, 15),
            n_comps=4,
            n_pcs=4
        )

        sc.tl.rank_genes_groups(adata, groupby="leiden")
        gc.collect()

        if return_copy:
            return adata
        else:
            self.adata = adata


    def plot_segmentation(
        self,
        plgns: gpd.GeoDataFrame | np.ndarray | None = None,
        channels: list[str | int, ...] | int | str = 0
    )-> Axes:
        """
        Plot the segmentation masks of `plgns` onto the region of interest.

        Parameters
        ----------
            plgns: geopandas.GeoDataFrame, numpy.ndarray or None
                The segmentation masks as either a geopandas.GeoDataFrame or numpy.ndarray.
                If not given, tries `plgns`.
                Optional.
            channels: list[str | int, ...], int or str
                Channels to use as background of the plot.
                Default is 0, i.e. the first channel.

        Returns
        -------
            out: tuple[matplotlib.figure.Figure, matplotlib.axis.Axes]
                The axis onto which was plotted.
        """
        if plgns is None:
            try:
                plgns = self.plgns
            except AttributeError:
                _plgns = self.find_cell_contour()
                plgns = self.plgns
        elif isinstance(plgns, np.ndarray):
            _plgns = self.find_cell_contour(masks=plgns)
            plgns = self.plgns
        else:
            raise TypeError(f'Not implemented to deal with {type(plgns)}.')

        if not isinstance(channels, list):
            try:
                channels = list(channels)
            except TypeError:
                channels = [channels]

        if isinstance(channels[0], int):
            pass
        elif isinstance(channels[0], str):
            channels = tuple(
                self.roi_biomarkers[c] for c in channels
            )
        else:
            raise TypeError(
                "`channels` has wrong type: \n"
                f"\t {type(channels[0])}"
            )

        _, ax = plt.subplots()
        ax = polygon_overlay(
            polygons=plgns,
            img=self.roi[channels, ...],
            ax=ax,
            pxsize=self.pxsize[0]
        )

        gc.collect()

        return ax


class IST:
    """
    imaging based spatially resolved transcriptomics
    """
    __slots__ = [
        '_roi',
        'path',
        'pxsize',
        'sections',
        'table',
        'z_bounds',
    ]
    def __init__(self, path, pxsize, sections):
        self.path = Path(path)
        self.pxsize = float(pxsize)
        self.sections = sections
        suffix = self.path.suffix
        if suffix:
            if suffix == '.parquet':
                self.table = pl.read_parquet(
                    self.path
                )
            else:
                raise NotImplementedError(
                    f'Not yet implemented to read {suffix} files.'
                )
        else:
            self.table = pl.read_parquet(
                self.path / 'transcripts.parquet'
            )

    @property
    def roi(self):
        """
        Sub-table with all transcripts in chosen bounds, and z-adjusted.
        """
        return self._roi

    @roi.setter
    def roi(self, coords,):
        new_coords = scale_coords(coords, self.pxsize)
        table = filter_by_location(
            self.table,
            new_coords,
        )
        table = adjust_z(table)
        self._roi = relative(
            table,
            new_coords
        )


class Phenium:
    """
    Class for combinging Xenium and PhenCycler and to deconvolute PhenoCycler.

    Parameters
    ----------
        SRT_dir :
            ...
        IHC_path :
            ...
        output_dir :
            ...
        config_path :
            ...
        machine : str, optional
            xenium
        prior_type : str, optional
            none, cellpose, proseg
        cell_size : int, optional
            ...
        seed : int, optional
            ...

    Attributes
    ----------
        prior_path
        layers
        CP_params
        PS_params
        SRT_section
        IHC_section
        transcripts
        scaling_factor
        seed
    """
    __slots__ = [
        "IHC",
        "IHC_path",
        "IST",
        "SRT_dir",
        "XMI",
        "_IHC_section",
        "_SRT_section",
        "_prior_path",
        "adata",
        "cell_size",
        "channels",
        "config",
        "counts",
        "decon_dir",
        "exclude",
        "machine",
        "masks",
        "output_dir",
        "prior_type",
        "scaling_factor",
        "seed",
        "sub_adata",
    ]
    def __init__(
        self,
        SRT_dir: str | PurePath,
        IHC_path: str | PurePath,
        output_dir: str | PurePath,
        config_path: str | PurePath,
        machine: str = 'xenium',
        prior_type: str = '',
        channels: tuple | None = None,
        exclude: list[(str | int), ...] | None = None,
        cell_size: int = 8,
        seed: int| None = None,
    )-> None:
        self.SRT_dir = Path(SRT_dir)
        self.IHC_path = Path(IHC_path)
        self.output_dir = Path(output_dir)

        Path(self.output_dir / 'plots/').mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = tomlkit.load(f)

        self.machine = machine
        self.prior_type = prior_type
        self.channels = channels
        self.exclude = exclude
        self.cell_size = cell_size
        self.seed = seed

        self.XMI = IMG(
            self.SRT_dir / 'morphology_focus/ch0000_dapi.ome.tif'
        )
        tifffile.imwrite(
            self.output_dir / 'SRT_overview.png',
            self.XMI.overview
        )

        self.IHC = IMG(self.IHC_path, exclude=exclude)
        tifffile.imwrite(
            self.output_dir / 'IHC_overview.png',
            self.IHC.overview
        )

        self.scaling_factor = (
            self.IHC.pxsize[0] / self.XMI.pxsize[0]
        )

        self.IST = IST(self.SRT_dir, self.XMI.pxsize[0], self.XMI.sections)

        save_settings(self, self.output_dir / 'settings.txt')

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            print(f'Set seed to {seed}')

        print(
            f'Initialized. \n'
            f'Review marked images in {self.output_dir}.'
        )


    @property
    def prior_path(self):
        """
        Path to the prior. Set after computing it.
        """
        try:
            return self._prior_path
        except AttributeError:
            return ''

    @prior_path.setter
    def prior_path(self, path: str | PurePath):
        self._prior_path = path

    @property
    def layers(self):
        """
        Layers defined in the transcriptomics by `cell_size`.
        """
        diff = (
            self.IST.roi['z_location'].max()
            - self.IST.roi['z_location'].min()
        )
        layers = int(diff/self.cell_size * 3)
        if layers > 3:
            return layers
        else:
            return 3

    @property
    def CP_params(self):
        """
        Parameters for the CellPose model and evaluation.
        """
        return self.config['cellpose']

    @property
    def PS_params(self):
        """
        Parameters for the ProSeg simulation.
        """
        return self.config['proseg']

    @property
    def SRT_section(self):
        """
        Section in the spatially resolved transcriptomics image.
        """
        try:
            return self._SRT_section
        except AttributeError:
            return None

    @SRT_section.setter
    def SRT_section(self, SRT_section):
        self._SRT_section = SRT_section

    @property
    def IHC_section(self):
        """
        Section in the immuno histo chemistry image.
        """
        try:
            return self._IHC_section
        except AttributeError:
            return None

    @IHC_section.setter
    def IHC_section(self, IHC_section):
        self._IHC_section = IHC_section

    def filter_for_section(
        self,
        table,
    ) -> pl.dataframe.frame.DataFrame:
        """
        Filter a table containing coordinates for points by a bounding box.
        The bounding box is defined by `SRT_section`.

        Parameters
        ----------
            table : polars DataFrame
                A DataFrame with columns containing coordinates of points.
                Either like `transcripts.parquet` or `cell_boundaries.parquet`
        
        Returns
        -------
            out : polars.dataframe.frame.DataFrame
                A polars dataframe, with entries all inside a defined bounding box.
        """
        new_coords = scale_coords(
            self.XMI.sections[self.SRT_section], self.XMI.pxsize[0]
        )
        filtered_table = filter_by_location(
            table,
            new_coords,
        )
        relative_filtered_table = relative(
            filtered_table,
            new_coords
        )
        gc.collect()
        return relative_filtered_table

    def _XOA_masks(self):
        """
        WIP as it is very slow.
        """
        path = self.SRT_dir / 'nucleus_boundaries.parquet'
        table = pl.read_parquet(path)
        ftable = self.filter_for_section(table)
        ptable = pixelate(ftable, self.XMI.pxsize[0])
        # print(table.head(n=5))
        masks = boundaries_to_mask(
            ptable,
            self.XMI.roi.shape[self.XMI.axes['Y']:self.XMI.axes['X']+1],
        )

        masks = masks.astype(np.uint32)

        self.prior_path = Path(
            self.output_dir / f'XOA_masks_{self.SRT_section}.npy'
        )

        np.save(
            self.prior_path,
            masks,
            allow_pickle=True
        )

        path = self.SRT_dir / 'cell_boundaries.parquet'
        table = pl.read_parquet(path)
        ftable = self.filter_for_section(table)
        ptable = pixelate(ftable, self.XMI.pxsize[0])
        masks = boundaries_to_mask(
            ptable,
            self.XMI.roi.shape[self.XMI.axes['Y']:self.XMI.axes['X']+1],
        )

        masks = masks.astype(np.uint32)

        self.masks = masks

        gc.collect()

    def prepare(
        self,
        SRT_section,
        IHC_section,
        skip_affine=False, # only for debug. remove later
        quick=False,
        verbose=False
    )-> None:
        """
        Prepare `Phenium` and the input for the Segmentation and Deconvolution.
         - sets the sections that correspond to each other, 
            - and aligns them;
         - sets the prior, if defined in the initialization;

        Parameters
        ----------
            SRT_section : int
                The number corresponding to the area of interest as depicted in the SRT_overview.
            IHC_section : int
                Same as `SRT_section` but based on the IHC_overview.
            quick : bool = False
                If `true` use the quick variant of the ants-registration algorithm.
            verbose : bool = False
                If `true` prints progress-bars for most functions.
        
        Returns
        -------
            out : None
                The results are assigned to object-variables.
        """
        if (
            (SRT_section in self.XMI.sections)
            and
            (IHC_section in self.IHC.sections)
        ):
            self.SRT_section = SRT_section
            self.IHC_section = IHC_section
        else:
            raise KeyError(f'This section (`{SRT_section}`) does not exist.')

        self.XMI.roi = SRT_section
        self.IHC.roi = IHC_section

        self.decon_dir = Path(
            self.output_dir / f'SRT{self.SRT_section}_IHC{self.IHC_section}/'
        ).mkdir(parents=True, exist_ok=True)

        if verbose:
            print('Filtering IST for section.')
        self.IST.roi = self.XMI.sections[SRT_section]

        IST_path = Path(
            self.output_dir / f'transcripts_{SRT_section}.parquet'
        )
        self.IST.roi.write_parquet(
            IST_path,
            compression="snappy",
            use_pyarrow=True,
            pyarrow_options={
                "version": "1.0",
            },
        )
        if verbose:
            print(f'Saved IST.roi at {IST_path}')

        if verbose:
            print('Adjusting IHC to be of same shape as SRT imgs')
        self.IHC.find_affine(
            self.XMI.roi[0, :, :],
            self.scaling_factor,
            quick=quick,
            verbose=verbose,
        )
        self.IHC.adjust(verbose=verbose)

        if self.prior_type == 'CP':
            style = self.CP_params['style']
            channels = ['DAPI']
            if style == 'focus':
                masks = self.XMI.segment(channels=channels, return_copy=True)
            elif style == 'pheno':
                masks = self.IHC.segment(channels=channels, return_copy=True)
            else:
                raise NotImplementedError(
                    'Not implemented to segment on'
                    f'{self.CP_params['style']}'
                )
            masks = masks.astype(np.uint32)
            self.prior_path = Path(
                self.output_dir / f"CP_masks_{self.SRT_section}_{style}.npy"
            )
            np.save(self.prior_path, masks, allow_pickle=True)
            self.masks = masks
        elif self.prior_type == 'XOA':
            _ = self._XOA_masks()
        else:
            print('No or unknown prior.')
            self.prior_path = ''

        print(
            f' Priors set to {self.prior_type}.'
            f'\n Prior_path: {self.prior_path}'
        )
        gc.collect()


    def run_proseg(
        self,
    ) -> None:
        """
        ...
        """
        proseg_wrapper(
            self.PS_params,
            self.machine,
            self.output_dir / f"transcripts_{self.SRT_section}.parquet",
            self.decon_dir,
            self.layers,
            self.prior_path,
        )

    def doublet_analysis(self):
        vrlp = ovrlpy_analysis(self.IST.roi, "xenium")

        doublets = vrlp.detect_doublets(min_signal=3, integrity_sigma=2)

        doublets = pixelate(doublets, self.XMI.pxsize[0])

        masks = load_plgns(Path(self.decon_dir) / "cell-polygon-layers.geojson.gz")
        masks = pixelate(masks, self.XMI.pxsize[0])

        _success_rate, stuff = vis_doublets(
            doublets,
            masks,
        )

        _other_stuff = proseg_doublets(masks)

        fig, ax = plt.subplots()
        fig, ax = polygon_overlay(
            masks[masks.layer == 0],
            self.XMI.roi[0, :, :],
            fig, ax,
            self.XMI.pxsize[0]
        )
        fig, ax = scatter_doublets(stuff, fig, ax, doublets)
        fig.savefig(self.decon_dir / "plots/proseg_L0_on_DAPI_wDoublets.pdf")


    def _load_counts(self):
        counts_path = self.decon_dir / 'counts.mtx.gz'
        counts = mmread(counts_path)
        self.counts = counts.tocsr()
        gc.collect()

    def _anndata(self):
        """
        Only here because loading the spatialdata obj didn't work.
        Hence, construction by hand.
        """
        self._load_counts()
        genes = pl.read_parquet(self.decon_dir / 'gene-metadata.parquet')
        cells = pl.read_parquet(self.decon_dir / 'cell-metadata.parquet')

        spatial = cells[
            ['centroid_x', 'centroid_y'] # , 'centroid_z']
        ].to_numpy()
        spatial_scaled = spatial * self.XMI.pxsize[0]

        obs = cells[['original_cell_id', 'cell']]
        obs_names = cells['original_cell_id']
        obsm = {'spatial' : spatial_scaled}

        var = genes[['gene', 'total_count', 'expected_assigned_count']]
        var_names = genes['gene']
        X = self.counts.tocsr()

        adata = ad.AnnData(
            X=X,
            obs=obs.to_pandas(),
            var=var.to_pandas(),
            obsm=obsm,
        )

        adata.obs_names = obs_names.to_pandas()
        adata.var_names = var_names.to_pandas()

        gc.collect()

        adata.write_h5ad(
            self.decon_dir / 'adata.h5ad'
        )

        return adata

    def cluster(
        self,
        resolution=0.25,
        save=True,
        return_copy=False,
    ):
        adata = self._anndata()

        adata = run_leiden(adata, resolution=resolution)

        sc.tl.rank_genes_groups(
            adata,
            groupby='leiden'
        )

        if save:
            save_anndata(adata, prefix=f'IST_{self.prior_type}', output_dir=self.decon_dir)

        gc.collect()

        if return_copy:
            return adata
        else:
            self.adata = adata

    def recluster(
        self,
        cluster,
        resolution,
        save=True,
        return_copy=False,
    ):
        adata = self.adata
        sub_adata = adata[adata.obs['leiden'].isin([str(cluster)]), :]
        sub_adata = run_leiden(sub_adata, resolution=resolution)
        sc.tl.rank_genes_groups(
            sub_adata,
            groupby='leiden'
        )
        if save:
            save_anndata(sub_adata, prefix=f'IST_c{cluster}', output_dir=self.decon_dir)

        gc.collect()

        if return_copy:
            return sub_adata
        else:
            self.sub_adata = sub_adata

    def rename_cluster(
        self,
        cluster_celltype: list | dict
    ):
        if isinstance(cluster_celltype, list):
            old_names = self.adata.obs['leiden']
            cluster_celltype = dict(zip(old_names, cluster_celltype))
        self.adata.obs["named_cluster"] = (
            self.adata.obs["leiden"].map(cluster_celltype).astype("category")
        )

    def assign_celltype(
        self,
        cluster_celltype: dict | None,
    ):
        gdf = load_plgns(self.decon_dir / "cell-polygon-layers.geojson.gz")

        if cluster_celltype is not None:
            self.rename_cluster(cluster_celltype)
            oci_leiden_dict = self.adata.obs.named_cluster.to_dict()       
        else:
            oci_leiden_dict = self.adata.obs.leiden()

        oci_ci_dict = self.adata.obs.cell.to_dict()            
        cellid_celltype_dict = {
            oci_ci_dict[oci]: oci_leiden_dict[oci] for oci in self.adata.obs_names
        }
        fltr = [idx in cellid_celltype_dict for idx in gdf.cell]
        gdf = gdf[fltr]
        gdf["celltype"] = gdf.apply(lambda row: cellid_celltype_dict[row.cell], axis=1)

        gc.collect()

        return gdf

    def deconvolute(
        self,
        cluster_celltype_dict: dict,
        celltype_prot_dict: dict,
        img: np.ndarray | None = None,
        biomarkers: dict | None = None,
        verbose: bool = False,
    ):
        if img is None:
            img = self.IHC.roi.view()
        elif isinstance(img, np.ndarray):
            img = img.view()
        else:
            raise TypeError(
                f'Given `img` is not `numpy.ndarray`, but {type(img)}'
            )

        if biomarkers is None:
            biomarkers = self.IHC.roi_biomarkers
        elif isinstance(biomarkers, dict):
            pass
        else:
            raise TypeError(
                f"`biomarkers` is not `dict`, but {type(biomarkers)}"
            )
        gdf = self.assign_celltype(cluster_celltype_dict, None)
        gdf = pixelate(gdf, self.IST.pxsize)
        if celltype_prot_dict is None:
            celltype_prot_dict = dict(
                zip(
                    [str(i) for i in range(len(self.adata.uns["leiden_colors"]) + 1)],
                    [
                        list(biomarkers)
                        for i in range(len(self.adata.uns["leiden_colors"]) + 1)
                    ],
                )
            )
        new_img = np.zeros(
            (
                self.layers,
                len(biomarkers),
            )
            + img.shape[~1:]
        )

        gc.collect()

        if verbose:
            pbar = tqdm.tqdm(total=len(gdf.index))
        for layer in range(gdf.layer.max()+1):
            for celltype in list(celltype_prot_dict.keys()):
                pbar.set_description(f'Processing L{layer}:CT{celltype}')
                prots = celltype_prot_dict[str(celltype)]
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
                fltr = (
                    (gdf.layer == layer).to_numpy() & 
                    (gdf.celltype == str(celltype)).to_numpy()
                )
                sub_gdf = gdf[fltr]
                # bnds = sub_gdf.bounds.astype(int)
                coords = sub_gdf.get_coordinates().astype(int)
                for loc in sub_gdf.index:
                    # minx, miny, maxx, maxy = bnds.loc[loc]
                    vrtcs = coords.loc[loc].to_numpy()
                    rr, cc = polygon(vrtcs[:,1], vrtcs[:,0])
                    for channel in channels:
                        new_img[layer, channel, rr, cc] = img[channel, rr, cc]
                    pbar.update(1)
        if verbose:
            pbar.close()

        # adjust for layer-frequency
        new_img = apply_factor(new_img, img)

        gc.collect()

        return new_img
