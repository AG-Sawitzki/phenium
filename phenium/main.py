from phenium.utils import (
    find_rois,
    bboxes_to_sections,
    filter_by_location,
    boundaries_to_mask,
    transcripts_for_ovrlpy,
    get_pixel_size,
    get_biomarkers,
    run_leiden,
)
from phenium.exceptions import (
    NotImplemented
)
from phenium.segment import (
    cellpose_wrapper,
    proseg_wrapper
)

import multiprocessing as mp
import os
import json
from pathlib import (
    PosixPath,
    Path
)
from typing import Union, Tuple

# import fast_matrix_market as fmm
import anndata as ad
import geopandas as gpd
import numpy as np
import polars as pl
import rapids_singlecell as rsc
import scanpy as sc
import squidpy as sq
from scipy.io import loadmat, mmread
from skimage.transform import rescale
from celltypist import models
import ants
import ovrlpy
import celltypist
import tifffile
import tomlkit
import zarr


class IMG:
    """
    multi-channel img
    (c, y, x)
    """
    def __init__(
        self,
        path,
    ):
        self.path = Path(path)
        store = tifffile.imread(self.path, aszarr=True)
        self.zrr = zarr.open(store, mode='r')
        self.shape = self.zrr['0'].shape
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
        return self._roi

    @roi.setter
    def roi(self, section):
        coords = self.sections[section]
        x = coords['x']
        y = coords['y']
        self._roi = self.zrr['0'][:, y[0]:y[1], x[0]:x[1]]
        ...

    @property
    def biomarkers(self):
        if self.path.suffix() == 'qptiff':
            biomarkers = get_biomarkers(self.path)
        else:
            biomarkers = {}
        
        return biomarkers


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
                    f'Not yet implemented to read {suffix} files'
                )
        else:
            self.table = pl.read_parquet(
                self.path / 'transcripts.parquet'
            )

        arr = self.table.select(
            pl.col('z_location')
        ).to_numpy()

        self.z_bounds = arr

    @property
    def z_bounds(self):
        return self._z_bounds

    @z_bounds.setter
    def z_bounds(self, arr):
        diff = int(max(arr)-min(arr))
        hist, edges = np.histogram(arr, bins=diff)
        elems = 0
        bins = []
        bins_by_size = np.argsort(hist)[::-1]
        for b in bins_by_size:
            bins.append(b)
            elems += hist[b]
            if elems/len(arr) >= 0.990:
                break

        self._z_bounds = edges[min(bins)], edges[max(bins)+1]

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
        cell_size: int, optional
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
    """
    def __init__(
        self,
        SRT_dir: Union[str, PosixPath],
        IHC_path: Union[str, PosixPath],
        output_dir: Union[str, PosixPath],
        config_path: Union[str, PosixPath],
        machine: str = 'xenium',
        prior_type: str = '',
        cell_size: int = 5,
    )-> None:
        self.SRT_dir = Path(SRT_dir)
        self.IHC_path = Path(IHC_path)
        self.output_dir = Path(output_dir)
    
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

        return None

    @property
    def prior_path(self):
        try:
            return self._prior_path
        except AttributeError:
            return ''

    @prior_path.setter
    def prior_path(self, path: Union[str, PosixPath]):
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

    def _filter_for_section(
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

    def _find_affine(
        self,
        quick=False,
        verbose=False
    ) -> None:
        SRT_img = self.SRT_img.roi[0, :, :]
        IHC_img = self.IHC_img.roi[0, :, :]
        IHC_img = rescale(
            IHC_img,
            self.scaling_factor,
            preserve_range=True,
        )
        IHC_img = np.flip(IHC_img)

        SRT_img = np.uint32(SRT_img)
        IHC_img = np.uint32(IHC_img)

        static = ants.from_numpy(SRT_img)
        moving = ants.from_numpy(IHC_img)

        if quick:
            tof = 'antsRegistrationSyNQuick[r]'
        else:
            tof = 'antsRegistrationSyN[r]'

        reg = ants.registration(
            fixed=static,
            moving=moving,
            type_of_transform=tof,
            verbose=verbose,
        )

        mat = loadmat(reg['fwdtransforms'][0])
        keys = list(mat.keys())

        dim = SRT_img.ndim
        A = mat[keys[0]][:dim**2].reshape((dim, dim))
        t = mat[keys[0]][dim**2:]
        c = mat[keys[1]]

        w = t + c - np.dot(A, c)
        Ah = np.identity(dim+1)
        Ah[:dim, :dim] = A
        Ah[:dim, dim:] = w

        self.affine_transform = Ah
        self.reg_fwd = reg['fwdtransforms']

    def _transform_IHC_channel(self, channel):
        img = self.IHC_img.roi[channel, :, :]
        img = rescale(
            img,
            self.scalgin_factor,
            preserve_range=True
        )
        img = np.flip(img)
        transformed_img = ants.apply_transforms(
            fixed=self.SRT_img.roi[0],
            moving=img,
            transformlist=self.reg_fwd,
            interpolator='nearestNeighbor',
        )

        return transformed_img

    def _XOA_masks(self):
        path = self.SRT_dir / 'cell_boundaries.parquet'
        table = pl.read_parquet(path)
        table = self._filter_for_section(table)
        # shape = self.get_shape(section)
        masks, polygons = boundaries_to_mask(table, self.SRT_img.roi.shape)
        
        self.prior_path = (
            self.output_dir / f'XOA_masks_{self.SRT_section}.npy'
        )

        np.save(
            self.prior_path,
            masks,
            allow_pickle=True
        )
        with open(self.output_dir / f'XOA_polygons_{self.SRT_section}.json') as f:
            json.dump(f, polygons)

    def _run_cellpose(self):
        model_params = self.CP_params['model']
        eval_params = self.CP_params['eval']
        style = self.CP_params['style']
        channels = self.CP_params['channels']

        img = 0

        if style == 'focus':
            img = self.SRT_img.roi[channels, :, :]
        if style == 'pheno':
            imgs = []
            for c in channels:
                imgs.append(self._transform_IHC_channel(c))
            img = np.dstack(imgs)
            img = np.moveaxis(img, -1, 0)

        if img:
            masks, _, __ = cellpose_wrapper(
                img,
                model_params,
                eval_params,
                style,
            )

            self.prior_path = (
                self.output_dir / f'CP_masks_{style}_{self.SRT_section}.npy'
            )

            np.save(
                self.prior_path,
                masks,
                allow_pickle=True
            )

    def prepare(
        self,
        SRT_section,
        IHC_section,
        skip_affine=False, # only for debug. remove later
        quick=True,
        verbose=True
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
            self._find_affine(quick=quick, verbose=verbose)

        if self.prior_type == 'CP':
            self._run_cellpose()
            # self.prior_path = self.output_dir / f'CP_masks_{SRT_section}.npy'
        elif self.prior_type == 'XOA':
            self._XOA_masks()
            # self.prior_path = self.output_dir / f'XOA_masks_{SRT_section}.npy'
        else:
            self.prior_path = ''

        print(
            f' Priors set to {self.prior_type}.'
            f'\n Prior_path: {self.prior_path}'
        )

    def calculate_VSI(self, transcripts, **kwargs):
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

    # ... more functions

    # def {function for running proseg, get_VSI, compare_VSI}
    #   assign transcripts to layers
    #
    #   get_VSI and compare_VSI must be optional...
    #   hence maybe separate into two functions

    # ... more functions

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

    def celltyping(self, resolution=0.25, rank_genes=True, gpu=True):
        self._anndata()

        self.adata = run_leiden(self.adata, resolution=resolution, gpu=gpu)

        sc.pl.umap(
            self.adata,
            color='leiden',
            legend_loc='on data',
            show=False,
        ).figure.savefig(
            self.output_dir / f'knn_umap_leiden_r{resolution}.png'
        )

        sc.tl.rank_genes_groups(
            self.adata,
            groupby='leiden'
        )

        self.adata.write_h5ad(
            self.output_dir / f'adata_r{resolution}.h5ad',
        )

        if rank_genes:
            rgg = self.adata.uns['rank_genes_groups']
            cluster = len(self.adata.uns['leiden_colors'])
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
                    (pl.col('pvals') < 0.01)
                ))
            pdf = pl.concat(temp_pdf_list)
            pdf.write_csv(self.output_dir / f'ranked_genes_r{resolution}.csv')

            sc.pl.rank_genes_groups_dotplot(
                self.adata,
                n_genes=3,
                values_to_plot='logfoldchanges',
                cmap='bwr',
                colorbar_title='log fold change',
                show=False,
                figsize = (12,6),
            )['mainplot_ax'].figure.savefig(
                self.output_dir / f'ranked_genes_dotplot_r{resolution}.png'
            )

        sq.pl.spatial_scatter(
            self.adata,
            shape=None,
            color='leiden',
            size=50,
            figsize=(50,50),
            save=(
                self.output_dir / f'spatial_scatter_leiden_r{resolution}.jpg'
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

        return self.adata

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

    def make_sparse_array(self, shape, layers):
        # it's not sparse rn. ik.
        shape = shape + (layers,)
        arr = np.zeros(shape)
        return arr

    # def assigne_cell(sparse_arr, plygn):
        # for channel in channels
            # cell_types = dict[channel] 
            # cells = all_cell[cell_type in cell_types]
            # 

    # def assign_CT_probabilities():

    # def assign_stain_to_CT():

    # def assign_pixel_to_cell(): why is this here?

    # def {function for running cluster_CellTypes, assign_CT, stain_to_Cell}
