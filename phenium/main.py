import gc
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain, product
from multiprocessing import Pool, cpu_count
from pathlib import Path, PurePath

import anndata as ad
import ants
import geopandas as gpd
import numpy as np
import polars as pl
import scanpy as sc
import squidpy as sq
import tifffile
import tomlkit
import torch
import tqdm
import zarr
from cellpose import models
from numpy.typing import ArrayLike
from scipy import stats
from scipy.io import loadmat, mmread
from scipy.sparse import csr_matrix
from shapely import transform
from shapely.geometry import Point
from skimage.transform import rescale

from phenium.segment import proseg_wrapper
from phenium.utils import (
    bboxes_to_sections,
    boundaries_to_mask,
    chunk_size,
    filter_by_location,
    find_rois,
    get_biomarkers,
    get_pixel_size,
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
        "feature_matrix",
        "masks",
        "overview",
        "path",
        "reg_factor",
        "reg_fwd",
        "reg_inv",
        "reg_shape",
        "sections",
        "zrr"
    ]
    def __init__(
        self,
        path,
    ):
        self.path = Path(path)
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
        return len(self.zrr)

    @property
    def shape(self):
        return self.zrr['0'].shape            

    @property
    def pxsize(self):
        return get_pixel_size(self.path)

    @property
    def roi(self):
        try:
            return self._roi_adj
        except AttributeError:
            return self._roi

    @roi.setter
    def roi(self, section):
        coords = self.sections[section]
        x = coords["x"]
        y = coords["y"]
        self._roi = self.zrr["0"][:, y[0]:y[1], x[0]:x[1]]

    @property
    def biomarkers(self):
        if self.path.suffix == ".qptiff":
            biomarkers = get_biomarkers(self.path)
        else:
            biomarkers = {}

        return biomarkers

    def find_affine(
        self,
        fixed: ArrayLike,
        scaling_factor: float,
        quick=False,
        verbose=False,
        matrix=False,
    ) -> None:
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
        verbose: bool = True,
        parallel: bool = True,
        return_copy: bool = False,
    ) -> None:
        """
        Adjusts the roi to fit the input image from `find_affine`.
        """
        iterations = list(self.biomarkers.values())
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
                        res.append(r)
            else:
                with Pool(processes=2) as pool:
                    res = pool.imap(func, moving)
                    pool.close()
                    pool.join()
            imgs = res
        else:
            if verbose:
                pbar = tqdm.tqdm(total=len(iterations))
            for channel in iterations:
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
        channels: list[int | str] | tuple[int | str],
        chunks: int = 4,
        return_copy: bool = False,
        verbose: bool = True,
        # mkwargs,
        # ekwargs
        gpu=True,
        pretrained_model='cpdino',
        device='cuda:0',
        use_bfloat16=True,
        batch_size=8,
        resample=True,
        channel_axis=None,
        cellprob_threshold=0.0,
        do_3D=False,
        flow3D_smooth=0,
        stitch_threshold=0.0,
        min_size=15,
        max_size_fraction=0.4,
        augment=False,
        tile_overlap=0.1,
        compute_masks=True
    ):
        if isinstance(channels[0], int):
            pass
        elif isinstance(channels[0], str):
            channels = tuple(
                self.biomarkers[c] for c in channels
            )
        else:
            raise TypeError(
                "`channels` has wrong type: \n"
                f"\t {type(channels[0])}"
            )

        roi = self.roi[channels, :, :].copy()

        _c, y, x = roi.shape

        x_size = chunk_size(x, chunks)
        y_size = chunk_size(y, chunks)

        grid = product(
            range(0, y-y % y_size, y_size),
            range(0, x-x % x_size, x_size)
        )

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
        for (y_c, x_c) in grid:
            y_low = int(y_c*0.95)
            y_high = int((y_c+y_size)*1.05)

            x_low = int(x_c * 0.95)
            x_high = int((x_c + x_size) * 1.05)

            roi_chunk = roi[
                :, y_low:y_high, x_low:x_high
            ]
            masks, _flows, _styles = model.eval(
                roi_chunk,
                resample=resample,
                channel_axis=channel_axis,
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
        gc.collect()

        if return_copy:
            return all_masks
        else:
            self.masks = all_masks       

    def make_feature_matrix(
        self,
        masks: ArrayLike=None,
        chunks: int = 4,
        parallel: bool=True,
        verbose: bool=True
    )-> None:
        if masks is None:
            masks = self.masks.copy()
        if masks.shape != self.roi.shape[1:]:
            self.adjust(verbose=verbose)

        cell_ids = np.unique(masks, sorted=True)[1:]
        cell_ids = np.astype(cell_ids, masks.dtype)
        cell_ids = np.expand_dims(cell_ids, 1)
        # feature_matrix = np.zeros((len(cell_ids), len(self.biomarkers)))
        roi = deepcopy(self.roi) if parallel else self.roi
        
        # _c, y, x = self.roi.shape
        # x_size = chunk_size(x, chunks)
        # y_size = chunk_size(y, chunks)
        # grid = product(
        #     range(0, y - y % y_size, y_size),
        #     range(0, x - x % x_size, x_size)
        # )
        datas = []
        arr_chunks = np.array_split(
            np.expand_dims(cell_ids, 1), chunks
        )

        func = partial(
            value_at_cell,
            masks = masks,
            roi = roi
        )
        cnuf = partial(
            np.apply_along_axis,
            func1d=func,
            axis=1
        )
        res = []
        with Pool(chunks) as pool:
            for r in tqdm.tqdm(pool.imap(cnuf, arr_chunks), total=chunks):
                res.append(r)
            pool.close()
            pool.join()
            print(res)
            res = np.concat(res)
            print(res)
        datas = res
        # for (y_c, x_c) in grid:
        #     y_low = int(y_c * 0.95)
        #     y_high = int((y_c + y_size) * 1.05)

        #     x_low = int(x_c * 0.95)
        #     x_high = int((x_c + x_size) * 1.05)

        #     roi_chunk = self.roi[:, y_low:y_high, x_low:x_high]
        #     mask_chunk = masks[y_low:y_high, x_low:x_high]

        #     func = partial(
        #         value_at_cell,
        #         masks=mask_chunk,
        #         roi=roi_chunk,
        #     )
        #     if parallel:
        #         res = []
        #         with Pool(processes=2) as pool:
        #             if verbose:
        #                 for r in tqdm.tqdm(
        #                     pool.imap(func, iterations),
        #                     total=len(iterations)
        #                 ):
        #                     res.append(r)
        #             else:
        #                 res = pool.imap(
        #                     func,
        #                     iterations
        #                 )
        #             pool.close()
        #             pool.join()
        #         print(res)
        #         data = np.concat(res)
        #         print(data)
        #     else:
        #         if verbose:
        #             pbar = tqdm.tqdm(total=len(cell_ids))
        #         for cell_id in cell_ids:
        #             vals = func(cell_id)
        #             # col.append(list(range(roi.shape[0])))
        #             data.append(vals)
        #             if verbose:
        #                 pbar.update(1)
        #         if verbose:
        #             pbar.close()
        #         data = np.array(data)
        #     datas.append(data)
        print(datas)
        # row = (cell_ids-1).tolist()
        # data = list(chain.from_iterable(data))
        # col = list(chain.from_iterable(col))
        feature_matrix = csr_matrix(datas)
        gc.collect()
        self.feature_matrix = feature_matrix

    def _anndata(self):
        """
        Making an anndata obj.
        """
        counts = self.feature_matrix

        obs = np.unique(self.masks, sorted=True)
        obs_names = np.unique(self.masks, sorted=True)

        var = self.biomarkers.keys()
        var_names = self.biomarkers.keys()
        X = counts.tocsr()

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
        )

        adata.obs_names = obs_names
        adata.var_names = var_names

        # library_id = f'tissue_{self.SRT_section}'
        # metadata = {
        #     'images': {'hires': self.XMI.roi[0, :, :]},
        #     'scalefactors': {
        #         'tissue_hires_scalef': 1,
        #         'spot_diameter_fullres': 0.25,
        #     },
        # }
        # adata.uns['spatial'] = {library_id: metadata}

        self.adata = adata

    def cluster(self, params, masks):
        self.segment(params)
        self.make_feature_matrix(masks)


class IST:
    """
    imaging based spatially resolved transcriptomics
    """
    __slots__ = [
        '_roi',
        'path',
        'pxsize',
        'table',
        'z_bounds',
    ]
    def __init__(self, path, pxsize):
        self.path = Path(path)
        self.pxsize = float(pxsize)
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

        self.adjust_z()

    @property
    def fov_names(self):
        return list(set(self.table['fov_name']))

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, coords,):
        new_coords = scale_coords(coords, self.pxsize)
        table = filter_by_location(
            self.table,
            new_coords,
            z_bounds=self.z_bounds,
        )
        self._roi = relative(
            table,
            new_coords
        )

    def adjust_z(self, return_copy=False):
        table = self.table
        x = table['x_location'].to_numpy()[::100]
        y = table['y_location'].to_numpy()[::100]
        z = table['z_location'].to_numpy()[::100]


        res_x = stats.linregress(x, z)
        res_y = stats.linregress(y, z)

        if res_x.slope < res_y.slope:
            res = res_y
            col = 'y_location'
        else:
            res = res_x
            col = 'x_location'

        for fov in self.fov_names:
            table_ = table.filter(pl.col('fov_name') == fov)
            exog = max(table_[col])

            table.with_columns(
                pl.when(
                    pl.col('fov_name') == fov
                ).then(
                    pl.col('z_location') - (res.intercept + res.slope*exog)
                ).otherwise(
                    pl.col('z_location')
                )
            )

        z_kde = stats.gaussian_kde(table['z_location'].to_numpy())

        z_min = table.select(pl.min('z_location'))
        z_max = table.select(pl.max('z_location'))

        z_space = np.linspace(z_min, z_max, 50).ravel()

        z_density = z_kde(z_space)

        z_density_argsort = np.argsort(z_density)[::-1]

        # perhaps isolation forest is the better idea
        # but what is a good contamination value?
        # auto did exclude too much...
        z_min = 0
        z_max = 0
        density_sum = 0
        for z_idx in z_density_argsort:
            dens = z_density[z_idx]
            z_val = z_space[z_idx]
            density_sum += dens
            if density_sum/sum(z_density) > 0.9:
                break
            z_min = min(z_min, z_val)
            z_max = max(z_max, z_max)
            if z_min == 0:
                z_min = z_val
            if z_max == 0:
                z_max = z_val

        expr = (
            pl.col('z_location').is_between(
                z_min, z_max, closed='both'
            )
        )
        table = table.filter(expr)

        table = table.with_columns(
            pl.col('z_location') - z_min
        )

        z = table.select('z_location').to_numpy()
        self.z_bounds = (z.min(), z.max())

        gc.collect()

        if return_copy:
            return table
        else:
            self.table = table
            return


class Phenium:
    """
    ...

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
        'adata',
        "cell_size",
        "config",
        'counts',
        "machine",
        "output_dir",
        "prior_type",
        "scaling_factor",
    ]
    def __init__(
        self,
        SRT_dir: str | PurePath,
        IHC_path: str | PurePath,
        output_dir: str | PurePath,
        config_path: str | PurePath,
        machine: str = 'xenium',
        prior_type: str = '',
        cell_size: int = 5,
        seed: int| None = None,
    )-> None:
        self.SRT_dir = Path(SRT_dir)
        self.IHC_path = Path(IHC_path)
        self.output_dir = Path(output_dir)

        Path(self.output_dir / 'plots/').mkdir(parents=False, exist_ok=True)

        with open(config_path) as f:
            self.config = tomlkit.load(f)

        self.machine = machine
        self.prior_type = prior_type
        self.cell_size = cell_size

        self.XMI = IMG(
            self.SRT_dir / 'morphology_focus/ch0000_dapi.ome.tif'
        )
        tifffile.imwrite(
            self.output_dir / 'SRT_overview.png',
            self.XMI.overview
        )

        self.IHC = IMG(self.IHC_path)
        tifffile.imwrite(
            self.output_dir / 'IHC_overview.png',
            self.IHC.overview
        )

        self.scaling_factor = (
            self.IHC.pxsize[0] / self.XMI.pxsize[0]
        )

        self.IST = IST(self.SRT_dir, self.XMI.pxsize[0])

        print(
            f'Initialized. \n'
            f'Review marked images in {self.output_dir}.'
        )

    @property
    def prior_path(self):
        try:
            return self._prior_path
        except AttributeError:
            return ''

    @prior_path.setter
    def prior_path(self, path: str | PurePath):
        self._prior_path = path

    @property
    def layers(self):
        z_bounds = self.IST.z_bounds
        diff = z_bounds[1]-z_bounds[0]
        layers = int(diff/self.cell_size * 3)
        if layers > 3:
            return layers
        else:
            return 3

    @property
    def CP_params(self):
        return self.config['cellpose']

    @property
    def PS_params(self):
        return self.config['proseg']

    @property
    def SRT_section(self):
        try:
            return self._SRT_section
        except AttributeError:
            return None

    @SRT_section.setter
    def SRT_section(self, SRT_section):
        self._SRT_section = SRT_section

    @property
    def IHC_section(self):
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
        # must be adapted for boundaries!
        # currently only transcripts are passed...
        filtered_table = filter_by_location(
            table,
            self.XMI.sections[self.SRT_section],
            pixelsize=self.XMI.pxsize[0]
        )
        gc.collect()
        return filtered_table

    def _XOA_masks(self):
        path = self.SRT_dir / 'cell_boundaries.parquet'
        table = pl.read_parquet(path)
        table = self.filter_for_section(table)
        print(table.head(n=5))
        masks = boundaries_to_mask(
            table,
            self.XMI.roi.shape,
            self.XMI.pxsize)

        self.prior_path = Path(
            self.output_dir / f'XOA_masks_{self.SRT_section}.npy'
        )

        np.save(
            self.prior_path,
            masks,
            allow_pickle=True
        )
        gc.collect()
        # with open(self.output_dir / f'XOA_polygons_{self.SRT_section}.json') as f:
        #     json.dump(f, polygons)

    def prepare(
        self,
        SRT_section,
        IHC_section,
        skip_affine=False, # only for debug. remove later
        quick=False,
        verbose=False
    ):
        '''
        Prepare `Phenium` and the input for the Segmentation and Deconvolution.
        '''
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

        if not skip_affine:
            if verbose:
                print('Adjusting IHC to be of same shape as SRT imgs')
            self.IHC.find_affine(
                self.XMI.roi[0, :, :],
                self.scaling_factor,
                quick,
                verbose,
            )
            self.IHC.adjust(verbose=verbose)

        # if self.prior_type == 'CP':
        #     style = self.CP_params['style']
        #     if style == 'focus':
        #         masks = self.XMI.segment(self.CP_params, return_copy=True)
        #     elif style == 'pheno':
        #         masks = self.IHC.segment(channels=channels, return_copy=True)
        #     else:
        #         raise NotImplementedError(
        #             'Not implemented to segment on'
        #             f'{self.CP_params['style']}'
        #         )
        #     self.prior_path = Path(
        #         self.output_dir / f"CP_masks_{self.SRT_section}_{style}.npy"
        #     )
        #     np.save(self.prior_path, masks, allow_pickle=True)
        # elif self.prior_type == 'XOA':
        #     self._XOA_masks()
        # else:
        #     print('No or unknown prior.')
        #     self.prior_path = ''

        print(
            f' Priors set to {self.prior_type}.'
            f'\n Prior_path: {self.prior_path}'
        )
        gc.collect()


    def _run_proseg(
        self,
    ) -> None:
        proseg_wrapper(
            self.PS_params,
            self.machine,
            self.output_dir / f'transcripts_{self.SRT_section}.parquet',
            self.output_dir,
            self.layers,
            self.prior_path
        )

    # def {function for running proseg, get_VSI}
        # indentify douplets and then investigate doublets
        # to show that proseg resolves or doesn't resolve those
        # doublets.

    def _load_counts(self):
        counts_path = self.output_dir / 'counts.mtx.gz'
        counts = mmread(counts_path)
        self.counts = counts.tocsr()

    def _anndata(self):
        """
        Only here because loading the spatialdata obj didn't work.
        Hence, construction by hand.
        """
        self._load_counts()
        genes = pl.read_parquet(self.output_dir / 'gene-metadata.parquet')
        cells = pl.read_parquet(self.output_dir / 'cell-metadata.parquet')

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

        # library_id = f'tissue_{self.SRT_section}'
        # metadata = {
        #     'images': {'hires': self.XMI.roi[0, :, :]},
        #     'scalefactors': {
        #         'tissue_hires_scalef': 1,
        #         'spot_diameter_fullres': 0.25,
        #     },
        # }
        # adata.uns['spatial'] = {library_id: metadata}

        self.adata = adata
        gc.collect()

    def rank_genes(
        self,
        resolution=0.25,
        gpu=False,
        return_copy=False,
        as_csv=True,
        as_dotplot=True,
        as_spatial_scatter=True,
    ):
        self._anndata()

        adata = self.adata

        adata = run_leiden(adata, resolution=resolution, gpu=gpu)

        sc.pl.umap(
            adata,
            color='leiden',
            legend_loc='on data',
            show=False,
        ).figure.savefig(
            self.output_dir / (
                f'plots/knn_umap_leiden_r{resolution}.pdf'
            )
        )

        sc.tl.rank_genes_groups(
            adata,
            groupby='leiden'
        )

        adata.write_h5ad(
            self.output_dir / f'adata_r{resolution}.h5ad',
        )

        if as_csv:
            rgg = adata.uns['rank_genes_groups']
            cluster = len(adata.uns['leiden_colors'])
            temp_pdf_list = []
            for c in range(cluster):
                temp_pdf = pl.DataFrame(
                    data={
                        'genes' : rgg['names'][str(c)],
                        'cluster': [c]*len(rgg['names'][str(c)]),
                        'pvals' : rgg['pvals_adj'][str(c)],
                        'logfc' : rgg['logfoldchanges'][str(c)],
                    }
                ).sort('pvals')
                temp_pdf_list.append(temp_pdf.filter(
                    pl.col('pvals') < 0.01
                ))
            pdf = pl.concat(temp_pdf_list)
            pdf.write_csv(self.output_dir / f'ranked_genes_r{resolution}.csv')

        if as_dotplot:
            sc.pl.rank_genes_groups_dotplot(
                adata,
                n_genes=3,
                values_to_plot='logfoldchanges',
                cmap='bwr',
                colorbar_title='log fold change',
                show=False,
                figsize = (12,6),
            )['mainplot_ax'].figure.savefig(
                self.output_dir / (
                    f'plots/ranked_genes_dotplot_r{resolution}.pdf'
                )
            )

        if as_spatial_scatter:
            sq.pl.spatial_scatter(
                adata,
                shape=None,
                color='leiden',
                size=50,
                figsize=(50,50),
                save=(
                    self.output_dir / (
                        f'plots/spatial_scatter_leiden_r{resolution}.pdf'
                    )
                )
            )

        # predictions = celltypist.annotate(
        #     self.adata,
        #     model=model,
        #     majority_voting=True,
        #     use_GPU=gpu
        # )

        # ctadata = predictions.to_adata()

        # sc.pl.umap(
        #     ctadata,
        #     color='predicted_labels',
        #     legend_loc='on data',
        #     show=False,
        # ).figure.savefig(self.output_dir / 'knn_umap_leiden_ct.png')

        gc.collect()

        if return_copy:
            return adata
        else:
            self.adata = adata

    # def rename_cluster(self, new_names):
    #     old_names = self.adata.obs['leiden']
    #     mapping_names = dict(
    #         zip(old_names, new_names)
    #     )
    #     self.adata.obs['named_cluster'] = (
    #         self.adata.obs['leiden'].map(
    #             mapping_names
    #         ).astype('category')
    #     )

    def function_name(
        self,
        cellid_celltype_dict,
        prot_celltype_dict,
    ):
        ply_path = self.output_dir / 'cell-polygon-layers.geojson.gz'
        gdf = gpd.read_file(
                ply_path.as_uri().replace('file', 'gzip')
            )

        if cellid_celltype_dict is None:
            oci_ci_dict = self.adata.obs.cell.to_dict()
            oci_leiden_dict = self.adata.obs.leiden.to_dict()
            cellid_celltype_dict = {
                oci_ci_dict[oci]: oci_leiden_dict[oci]
                for oci in self.adata.obs_names
            }
        fltr = [
            idx in cellid_celltype_dict for idx in gdf.cell
        ]
        gdf = gdf[fltr]
        gdf['celltype'] = gdf.apply(
            lambda row: cellid_celltype_dict[row.cell],
            axis=1
        )

        if prot_celltype_dict is None:
            prot_celltype_dict = dict(zip(
                self.IHC.biomarkers,
                np.random.randint(
                    0,
                    len(self.adata.uns['leiden_colors']),
                    size=len(self.IHC.biomarkers))
            ))

        new_img = np.zeros(
            (len(self.IHC.biomarkers), self.layers,)
            + self.XMI.roi.shape[1:]
        )

        gc.collect()

        for prot, celltypes in prot_celltype_dict.items():
            index = self.IHC.biomarkers[prot]
            channel = self.IHC.roi[index, :, :]
            if not isinstance(celltypes, list | tuple):
                celltypes = [celltypes]
            for celltype in celltypes:
                cells = gdf.loc[gdf.celltype == celltype]
                coords = np.argwhere(channel > 0)
                for coord in coords:
                    pnkt = transform(Point(coord), lambda x: x/self.XMI.pxsize[0])
                    mask = cells.contains(pnkt)
                    if mask.any():
                        layers = set(cells.loc[cells.loc[mask]]['layer'])
                        new_value = channel[tuple(coord)]/len(layers)
                        new_img[index, tuple(layers), coord] = new_value

        gc.collect()

        return new_img
