from phenium.utils import (
    find_rois,
    bboxes_to_sections,
    filter_by_location,
    boundaries_to_mask,
    transcripts_for_ovrlpy,
    # find_transform,
    get_pixel_size,
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
from scipy.io import loadmat, mmread
from skimage.transform import rescale
from celltypist import models
import ants
import ovrlpy
import celltypist
import tifffile
import tomlkit
import zarr



class Phenium(object):
    def __init__(
        self,
        SRT_dir,
        IHC_path,
        output_dir,
        config_path,
    )-> None:
        self.SRT_dir = Path(SRT_dir)
        self.IHC_path = Path(IHC_path)
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)

        with open(self.config_path) as f:
            self.config = tomlkit.load(f)

        self._find_sections()

        print('Initialized. Review marked images and provide sections.')

    @property
    def machine(self):
        return self.config['Basics']['machine']

    @property
    def priors_type(self):
        return self.config['Basics']['priors']

    @property
    def prior_path(self):
        return self._prior_path

    @prior_path.setter
    def prior_path(self, path: Union[str, PosixPath]):
        self._prior_path = path

    @property
    def layers(self):
        cell_size = self.config['Basics']['size']
        diff = self.z_bounds[1]-self.z_bounds[0]
        layers = int(diff/cell_size * 3)
        if layers > 3:
            return layers
        else:
            return 3

    @property
    def celltypist_model(self):
        model = self.config['Basics']['CT_model']
        return  model

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

    @SRT_section.setter
    def IHC_section(self, IHC_section):
        self._IHC_section = IHC_section

    @property
    def transcripts(self):
        transcripts = pl.read_parquet(
            self.SRT_dir / 'transcripts.parquet'
        )

        if self.SRT_section != None:
            transcripts_filterd = self._filter_for_section(
                transcripts,
            )
            return transcripts_filterd
        else:
            return transcripts

    @property
    def z_bounds(self):
        table = self.transcripts
        table = table.rename(ovrlpy.io._10X_GENOMICS_COLUMNS)
        table = ovrlpy.io._filter_genes(table, ovrlpy.io.XENIUM_CTRLS)
        arr = table['z'].to_numpy()
        diff = int(max(arr)-min(arr))
        hist, edges = np.histogram(arr, bins=diff)
        elems = 0
        bins = []
        i = 0
        while elems/len(arr) < 0.990:
            bins_by_size = np.argsort(hist)[::-1]
            bin_id = bins_by_size[i]
            elems+=hist[bin_id]
            bins.append(bin_id)
            i+=1

        return edges[min(bins)], edges[max(bins)+1]

    @property
    def transcripts_path(self):
        transcripts_path = Path(
            self.output_dir / f'transcripts_{self.SRT_section}.parquet'
        )
        return transcripts_path

    @property
    def SRT_zarr(self):
        SRT_store = tifffile.imread(
            self.SRT_dir / 'morphology_focus/ch0000_dapi.ome.tif',
            aszarr=True
        )
        zrr = zarr.open(SRT_store, mode='r')
        return zrr

    @property
    def SRT_shape(self):
        return self.SRT_zarr['0'].shape

    @property
    def SRT_pxsize(self):
        pxsize = get_pixel_size(
            self.SRT_dir / 'morphology_focus/ch0000_dapi.ome.tif'
        )
        return pxsize

    @property
    def IHC_zarr(self):
        IHC_store = tifffile.imread(
            self.IHC_path,
            aszarr=True
        )
        zrr = zarr.open(IHC_store, mode='r')
        return zrr

    @property
    def IHC_shape(self):
        return self.IHC_zarr['0'].shape

    @property
    def IHC_pxsize(self):
        pxsize = get_pixel_size(
            self.IHC_path
        )
        return pxsize

    @property
    def scaling_factor(self):
        factor = self.IHC_pxsize[0]/self.SRT_pxsize[0]
        return factor

    @property
    def SRT_sections(self):
        return self._SRT_sections

    @SRT_sections.setter
    def SRT_sections(self, sections: dict):
        self._SRT_sections = sections

    @property
    def IHC_sections(self):
        return self._IHC_sections

    @IHC_sections.setter
    def IHC_sections(self, sections: dict):
        self._IHC_sections = sections

    @property
    def SRT_coords(self):
        return self.SRT_sections[self.SRT_section]

    @property
    def IHC_coords(self):
        return self.IHC_sections[self.IHC_section]

    @property
    def SRT_img(self):
        img = self.SRT_zarr['0'][
            :,
            self.SRT_coords['y'][0]:self.SRT_coords['y'][1],
            self.SRT_coords['x'][0]:self.SRT_coords['x'][1]
        ]
        return img

    @property
    def section_shape(self):
        return self._get_shape()

    def _find_sections(
        self
    ) -> None:
        subresolution_lvl = len(self.SRT_zarr)
        bboxes = find_rois(
            self.SRT_shape,
            self.SRT_zarr[str(subresolution_lvl-2)]
        )
        self.SRT_sections, marked_img = bboxes_to_sections(
            bboxes,
            self.SRT_shape,
            self.SRT_zarr[str(subresolution_lvl-2)]
        )

        tifffile.imwrite(
            self.output_dir / 'marked_regions_SRT.tif',
            marked_img
        )

        subresolution_lvl = len(self.IHC_zarr)
        bboxes = find_rois(
            self.IHC_shape,
            self.IHC_zarr[str(subresolution_lvl-1)]
        )
        self.IHC_sections, marked_img = bboxes_to_sections(
            bboxes,
            self.IHC_shape,
            self.IHC_zarr[str(subresolution_lvl-1)]
        )

        tifffile.imwrite(
            self.output_dir / 'marked_regions_IHC.tif',
            marked_img
        )

    def _get_shape(
        self,
    ) -> Tuple[int, int]:
        x = self.SRT_coords['x'][1]-self.SRT_coords['x'][0]
        y = self.SRT_coords['y'][1]-self.SRT_coords['y'][0]

        return x, y

    def _filter_for_section(
        self,
        table,
    ) -> pl.dataframe.frame.DataFrame:
        # must be adapted for boundaries!
        # currently only transcripts are passed...
        filtered_table = filter_by_location(
            table,
            self.SRT_coords,
            self.z_bounds,
            pixelsize=self.SRT_pxsize[0]
        )

        return filtered_table

    def _find_affine(
        self,
        quick=False,
        verbose=False
    ) -> None:
        SRT_img = self.SRT_img[0, :, :]
        IHC_img = self.IHC_zarr['0'][
            0,
            self.IHC_coords['y'][0]:self.IHC_coords['y'][1],
            self.IHC_coords['x'][0]:self.IHC_coords['x'][1]
        ]
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
        img = self.IHC_zarr['0'][
            channel,
            self.IHC_coords['y'][0]:self.IHC_coords['y'][1],
            self.IHC_coords['x'][0]:self.IHC_coords['x'][1]
        ]
        transformed_img = ants.apply_transforms(
            fixed=self.SRT_img[0],
            moving=img,
            transformlist=self.reg_fwd,
            interpolator='nearestNeighbor',
        )

        return transformed_img

    def _XOA_masks(self, section):
        path = self.SRT_dir / 'cell_boundaries.parquet'
        table = pl.read_parquet(path)
        table = self.filter_for_section(section, table)
        # shape = self.get_shape(section)
        masks, polygons = boundaries_to_mask(table, self.roi_shape)
        np.save(
            self.output_dir / f'XOA_masks_{section}.npy',
            masks,
            allow_pickle=True
        )
        with open(self.output_dir / f'XOA_polygons_{section}.json') as f:
            json.dump(f, polygons)

    def _run_cellpose(self):
        model_params = self.CP_params['model']
        eval_params = self.CP_params['eval']
        style = self.CP_params['style']
        channels = self.CP_params['channels']

        if style == 'focus':
            zrr = self.SRT_zarr
            img = zrr['0'][
                channels,
                self.SRT_coords['y'][0]:self.SRT_coords['y'][1],
                self.SRT_coords['x'][0]:self.SRT_coords['x'][1]
            ]
        if style == 'pheno':
            imgs = []
            for c in channels:
                imgs.append(self._transform_IHC_channel(c))
            img = np.dstack(imgs)
            img = np.moveaxis(img, -1, 0)

        masks, flows, styles = cellpose_wrapper(
            img,
            model_params,
            eval_params,
            style,
        )

        np.save(
            self.output_dir / f'CP_masks_{self.SRT_section}.npy',
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
            (SRT_section in self.SRT_sections.keys())
            and
            (IHC_section in self.IHC_sections.keys())
        ):
            self.SRT_section = SRT_section
            self.IHC_section = IHC_section
        else:
            # this error isn't returned
            KeyError(f'This section (`{SRT_section}`) does not exist.')
            return None
        
        self.transcripts.write_parquet(
            self.transcripts_path,
            compression='snappy',
            use_pyarrow=True,
            pyarrow_options=dict(
                version = '1.0',
            )
        )

        if not skip_affine:
            self._find_affine(quick=quick, verbose=verbose)

        if self.priors_type == 'CP':
            self._run_cellpose(
                SRT_section
            )
            self.prior_path = self.output_dir / f'CP_masks_{SRT_section}.npy'
        elif self.priors_type == 'XOA':
            self._XOA_masks(
                SRT_section
            )
            self.prior_path = self.output_dir / f'XOA_masks_{SRT_section}.npy'
        else:
            self.prior_path = ''

        print(
            f' Priors set to {self.priors_type}.'
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
            self.transcripts_path,
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

        obs = cells[['original_cell_id', 'cell']]
        obs_names = cells['original_cell_id']
        obsm = dict(
            spatial = cells[
                ['centroid_x', 'centroid_y', 'centroid_z']
            ].to_numpy()
        )

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

        self.adata = adata

    def celltyping(self):
        self._anndata()

        self.adata = run_leiden(self.adata, gpu=True)

        sc.pl.umap(
            self.adata,
            color='leiden',
            legend_loc='on data',
            show=False,
        ).figure.savefig(self.output_dir / 'knn_umap_leiden_gpu.png')

        # models.download_models(force_update = True)

        try:
            model = models.Model.load(
                model=self.celltypist_model
            )
        except FileNotFoundError:
            print('No such model. Perhaps you mistyped.')
            print(f'{self.celltypist_model}')
            return None

        predictions = celltypist.annotate(
            self.adata,
            model = self.celltypist_model,
            majority_voting=True
        )

        adata = predictions.to_adata()

        sc.pl.umap(
            adata,
            color='predicted_labels',
            legend_loc='on data',
            show=False,
        ).figure.savefig(self.output_dir / 'knn_umap_leiden_ct_gpu.png')

        return adata

    # def make_sparse_array(img.shape + (proseg_layer,)):

    # def assigne_cell(sparse_arr, plygn):
        
    # def assign_CT_probabilities():

    # def assign_stain_to_CT():

    # def assign_pixel_to_cell(): why is this here?

    # def {function for running cluster_CellTypes, assign_CT, stain_to_Cell}