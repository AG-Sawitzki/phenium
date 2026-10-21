import json
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path, PurePath

import anndata as ad
import ants
import geopandas as gpd
import numpy as np
import ovrlpy
import polars as pl
import scanpy as sc
import squidpy as sq
import tifffile
import tomlkit
import tqdm
import zarr
from numpy.typing import ArrayLike
from scipy import stats
from scipy.io import loadmat, mmread
from shapely import transform
from shapely.geometry import Point
from skimage.transform import rescale

from phenium.segment import cellpose_wrapper, proseg_wrapper
from phenium.utils import (
    bboxes_to_sections,
    boundaries_to_mask,
    chunk_size,
    filter_by_location,
    find_rois,
    get_biomarkers,
    get_pixel_size,
    run_leiden,
    transcripts_for_ovrlpy,
)


class IMG:
    """
    multi-channel img
    (c, y, x) or (c, z, y, x)
    """
    def __init__(
        self,
        path,
    ):
        self.path = Path(path)
        store = tifffile.imread(self.path, aszarr=True)
        self.zrr = zarr.open(store, mode="r")
        self.shape = self.zrr["0"].shape
        self.pxsize = get_pixel_size(self.path)

        subresolution_lvl = len(self.zrr)
        bboxes = find_rois(
            self.shape,
            self.zrr[str(subresolution_lvl-2)]
        )
        self.sections, self.overview = bboxes_to_sections(
            bboxes,
            self.shape,
            self.zrr[str(subresolution_lvl-2)]
        )

    @property
    def roi(self):
        try:
            return self._roi_adj
        except AttributeError:
            return self._roi

    @roi.setter
    def roi(self, section,
            # scaling_factor=None
    ):
        coords = self.sections[section]
        x = coords["x"]
        y = coords["y"]
        self._roi = self.zrr["0"][:, y[0] : y[1], x[0] : x[1]]

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

        static = ants.from_numpy(fixed)
        moving = ants.from_numpy(moving)

        if quick:
            tof = "antsRegistrationSyNQuick[r]"
        else:
            tof = "antsRegistrationSyN[r]"

        reg = ants.registration(
            fixed=static,
            moving=moving,
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

    def transform_channel(self, channel, fixed, scaling_factor):
        if isinstance(channel, str):
            channel = self.biomarkers[channel]

        moving = self.roi[channel]
        moving = rescale(
            moving,
            scaling_factor,
            preserve_range=True,
        )
        fixed = np.int32(fixed)
        moving = np.int32(moving)

        fixed_ant = ants.from_numpy(fixed)
        moving_ant = ants.from_numpy(moving)

        moving_transformed = ants.apply_transforms(
            fixed=fixed_ant,
            moving=moving_ant,
            transformlist=self.reg_fwd,
            interpolator='nearestNeighbor',
        )

        return moving_transformed

    def adjust(
        self,
        fixed: ArrayLike,
        scaling_factor: float,
        quick: bool = False,
        verbose: bool = False,
        parallelize: bool = False,
        return_copy: bool = False,
    ) -> None:
        """
        Adjusts the roi to fit the input image from `find_affine`.
        Replaces `self.roi`.
        """
        self.find_affine(fixed, scaling_factor, quick, verbose)
        imgs = []

        if parallelize:
            pool = Pool(processes=int(cpu_count()/2))
            func = partial(
                self.transform_channel,
                fixed=fixed,
                scaling_factor=scaling_factor,
            )
            iterations = self.biomarkers.values()
            if verbose:
                res = list(tqdm.tqdm(
                    pool.imap_unordered(func, iterations),
                    total=len(iterations),
                ))
                pool.join()
                pool.close()
                print(res)
                imgs.append(res)
            else:
                res = pool.imap_unordered(func,iterations)
                pool.join()
                pool.close()
                print(res)
                imgs = list(res)
            print(imgs)
        else:
            if verbose:
                pbar = tqdm.tqdm(total=len(self.biomarkers))
            for channel in self.biomarkers.values():
                img = self.transform_channel(
                    channel=channel,
                    fixed=fixed,
                    scaling_factor=scaling_factor
                )
                # img_c = img[channel]
                # if scaling_factor is not None:
                #     img_c = rescale(
                #         img_c,
                #         scaling_factor,
                #         preserve_range=True,
                #     )
                # fixed = np.int32(fixed)
                # moving = np.int32(img_c)
                # fixed_ant = ants.from_numpy(fixed)
                # moving_ant = ants.from_numpy(moving)
                # img_c = ants.apply_transforms(
                #     fixed=fixed_ant,
                #     moving=moving_ant,
                #     transformlist=self.reg_fwd,
                #     interpolator="nearestNeighbor",
                # )
                imgs.append(img.numpy())
                if verbose:
                    pbar.update(1)
        if verbose:
            pbar.close()
            
        imgs_stacked = np.dstack(imgs)
        imgs_stacked_moved = np.moveaxis(imgs_stacked, -1, 0)
        self._roi_adj = imgs_stacked_moved
        if return_copy:
            return imgs_stacked_moved

    def segment(
        self,
        params: dict | tomlkit.items.Table,
        return_copy: bool = False
    ):
        channels = tuple(params['channels'])
        chunks = params['chunks']
        eval_params = params['eval']
        model_params = params['model']
        
        if isinstance(channels[0], int):
            pass
        elif isinstance(channels[0], str):
            channels = tuple(
                self.biomarkers[c] for c in channels
            )
        else:
            raise TypeError(
                "`channels` in `params` has wrong type: \n"
                f"\t {type(channels)}"
            )

        _c, y, x = self.roi.shape

        x_size = chunk_size(x, chunks)
        y_size = chunk_size(y, chunks)

        grid = product(
            range(0, y-y % y_size, y_size),
            range(0, x-x % x_size, x_size)
        )

        all_masks = np.zeros(self.roi.shape[1:])

        for chunk, (y_c, x_c) in enumerate(grid):
            y_low = int(y_c*0.95)
            y_high = int((y_c+y_size)*1.05)

            x_low = int(x_c * 0.95)
            x_high = int((x_c + x_size) * 1.05)

            roi_chunk = self.roi[
                channels, y_low:y_high, x_low:x_high
            ]

            masks, _, __ = cellpose_wrapper(
                roi_chunk,
                model_params,
                eval_params,
            )

            all_masks[y_low:y_high, x_low:x_high] = masks

        self.masks = all_masks
        if return_copy:
            return all_masks

    def make_feature_matrix(self, masks,):
        cell_ids = np.unique(masks, sorted=True)
        feature_matrix = np.zeros((len(cell_ids), len(self.biomarkers)))
        for cell_id in cell_ids:
            loc = np.argwhere(self.masks == cell_id)
            y_loc = tuple(loc[:, 0])
            x_loc = tuple(loc[:, 1])
            for channel in self.biomarkers.values():
                val = np.sum(self.roi[channel, y_loc, x_loc])
                feature_matrix[cell_id, channel] = val
        self.feature_matrix = feature_matrix

    def cluster(self, params, masks):
        self.segment(params)
        self.make_feature_matrix(masks)


        


class IST:
    """
    imaging based spatially resolved transcriptomics
    """
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
        self._roi = filter_by_location(
            self.table,
            coords,
            z_bounds=self.z_bounds,
            pixelsize=self.pxsize
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

        self.SRT_img = IMG(
            self.SRT_dir / 'morphology_focus/ch0000_dapi.ome.tif'
        )
        tifffile.imwrite(
            self.output_dir / 'SRT_overview.png',
            self.SRT_img.overview
        )

        self.IHC_img = IMG(self.IHC_path)
        tifffile.imwrite(
            self.output_dir / 'IHC_overview.png',
            self.IHC_img.overview
        )

        self.scaling_factor = (
            self.IHC_img.pxsize[0] / self.SRT_img.pxsize[0]
        )

        self.IST = IST(self.SRT_dir, self.SRT_img.pxsize[0])

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
            self.SRT_img.sections[self.SRT_section],
            pixelsize=self.SRT_img.pxsize[0]
        )

        return filtered_table

    def _XOA_masks(self):
        path = self.SRT_dir / 'cell_boundaries.parquet'
        table = pl.read_parquet(path)
        table = self.filter_for_section(table)
        print(table.head(n=5))
        masks = boundaries_to_mask(
            table,
            self.SRT_img.roi.shape,
            self.SRT_img.pxsize)

        self.prior_path = Path(
            self.output_dir / f'XOA_masks_{self.SRT_section}.npy'
        )

        np.save(
            self.prior_path,
            masks,
            allow_pickle=True
        )
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
            (SRT_section in self.SRT_img.sections)
            and
            (IHC_section in self.IHC_img.sections)
        ):
            self.SRT_section = SRT_section
            self.IHC_section = IHC_section
        else:
            raise KeyError(f'This section (`{SRT_section}`) does not exist.')

        self.SRT_img.roi = SRT_section
        self.IHC_img.roi = IHC_section

        self.IST.roi = self.SRT_img.sections[SRT_section]

        self.IST.roi.write_parquet(
            self.output_dir / f'transcripts_{SRT_section}.parquet',
            compression='snappy',
            use_pyarrow=True,
            pyarrow_options={
                'version' : '1.0',
            }
        )

        if not skip_affine:
            self.IHC_img.adjust(
                self.SRT_img.roi[0, :, :],
                self.scaling_factor,
                quick=quick,
                verbose=verbose,
                parallelize=True,
            )

        if self.prior_type == 'CP':
            style = self.CP_params['style']
            if style == 'focus':
                masks = self.SRT_img.segment(self.CP_params, return_copy=True)
            elif style == 'pheno':
                masks = self.IHC_img.segment(self.CP_params, return_copy=True)
            else:
                raise NotImplementedError(
                    'Not implemented to segment on'
                    f'{self.CP_params['style']}'
                )
            self.prior_path = Path(
                self.output_dir / f"CP_masks_{self.SRT_section}_{style}.npy"
            )
            np.save(self.prior_path, masks, allow_pickle=True)
        elif self.prior_type == 'XOA':
            self._XOA_masks()
        else:
            print('No or unknown prior.')
            self.prior_path = ''

        print(
            f' Priors set to {self.prior_type}.'
            f'\n Prior_path: {self.prior_path}'
        )

    def calculate_VSI(self, transcripts, plot=True, **kwargs):
        cdf = transcripts_for_ovrlpy(
            machine=self.machine,
            transcripts=transcripts,
        )

        ovrlpy_dataset = ovrlpy.Ovrlp(
            cdf,
            # KDE_bandwidth,
            # min_distance,
            # n_components,
            # gene_key,
            # coordinate_keys,
            # n_workers,
            # dtype,
            # patch_length,
            # umap_kwargs,
            # cumap_kwargs,
            # random_state,
        )

        # should be split into the processes
        # to include all params for
        # process_coordinates
        # fit_transcripts
        # compute_VSI
        ovrlpy_dataset.analyse(
            # gridsize,
            # min_transcripts,
            # genes,
            # fit_umap,
        )

        if plot:
            fig = ovrlpy.plot_signal_integrity(
                ovrlpy_dataset,
                signal_threshold=4,
            )
            fig.savefig(
                self.output_dir / 'plots/signal_integrity_map.pdf'
            )

            fig = ovrlpy.plot_pseudocells(ovrlpy_dataset)
            fig.savefig(
                self.output_dir / 'plots/pseudocells.pdf'
            )

        return ovrlpy_dataset.integrity_map, ovrlpy_dataset.signal_map

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
        spatial_scaled = spatial * self.SRT_img.pxsize[0]

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
        #     'images': {'hires': self.SRT_img.roi[0, :, :]},
        #     'scalefactors': {
        #         'tissue_hires_scalef': 1,
        #         'spot_diameter_fullres': 0.25,
        #     },
        # }
        # adata.uns['spatial'] = {library_id: metadata}

        self.adata = adata

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

        if return_copy:
            return adata
        else:
            self.adata = adata
            return

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
        gdf['celltype'] = gdf.apply(
            lambda row: cellid_celltype_dict[row.cell],
            axis=1
        )
        c = len(self.IHC_img.biomarkers)
        z = self.layers
        y, x = self.SRT_img.shape[1:]
        new_img = np.zeros((c, z, y, x))
        for prot, celltypes in prot_celltype_dict:
            for celltype in celltypes:
                index = self.IHC_img.biomarkers[prot]
                channel = self.IHC_img.roi[index, :, :]
                # channel = self.transform_IHC_channel(self.IHC_img[index, :, :])
                cells = gdf.loc[gdf.celltype == celltype]
                coords = np.argwhere(channel > 0)
                for coord in coords:
                    point = transform(Point(coord), lambda x: x/self.SRT_img.pxsize[0])
                    mask = cells.contains(point)
                    if mask.any():
                        layers = set(cells.loc[cells.loc[mask]]['layer'])
                        new_value = channel[tuple(coord)]/len(layers)
                        new_img[index, tuple(layers), coord] = new_value

            return new_img
